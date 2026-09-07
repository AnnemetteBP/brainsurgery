#!/usr/bin/env python3
"""Run the original paper's behavioral metrics on the sourced 70-prompt suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers

from revision_tests.behavioral.run_model import (
    config_fingerprint,
    read_manifest,
    score_choices,
    tokenizer_fingerprint,
)
from validation.test_inference import (
    _encode_prompt,
    _full_sequence_logit_metrics,
    _generate_ids,
    _load_tokenizer,
    _output_similarity_metrics,
    _tokenizer_label,
    load_model,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "prompt_manifest.jsonl"
PROTOCOL_ID = "eacl2027_behavioral_paper_v4"
REQUIRED_PROMPT_METRICS = {
    "reference_perplexity",
    "transformed_perplexity",
    "perplexity_ratio",
    "last_token_logit_cosine",
    "last_token_mean_absolute_difference",
    "last_token_max_absolute_difference",
    "top1_match",
    "full_sequence_positions_compared",
    "full_sequence_mean_logit_cosine",
    "full_sequence_min_logit_cosine",
    "full_sequence_mean_abs_logit_diff",
    "full_sequence_max_abs_logit_diff",
    "output_exact_match",
    "output_char_similarity",
    "output_token_sequence_similarity",
    "output_token_bag_cosine",
    "mcq_prediction_match",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--transformed", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), required=True
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.01)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--smoke-limit", type=int)
    return parser.parse_args()


def git_value(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_complete_rows(rows: list[dict[str, Any]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} prompt results, found {len(rows)}")
    for row in rows:
        missing = REQUIRED_PROMPT_METRICS - row.keys()
        if missing:
            raise ValueError(f"{row.get('prompt_id', '<unknown>')}: missing {sorted(missing)}")
        for key in REQUIRED_PROMPT_METRICS - {"top1_match", "output_exact_match", "mcq_prediction_match"}:
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{row['prompt_id']}: {key} is not finite")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require_complete_rows(rows, len(rows))
    count = len(rows)
    mcq_rows = [row for row in rows if row["expected_kind"] == "multiple_choice"]
    return {
        "prompt_count": count,
        "reference_mean_perplexity": sum(row["reference_perplexity"] for row in rows) / count,
        "transformed_mean_perplexity": sum(row["transformed_perplexity"] for row in rows)
        / count,
        "mean_perplexity_ratio": sum(row["perplexity_ratio"] for row in rows) / count,
        "maximum_perplexity_ratio": max(row["perplexity_ratio"] for row in rows),
        "mean_last_token_logit_cosine": sum(row["last_token_logit_cosine"] for row in rows)
        / count,
        "minimum_last_token_logit_cosine": min(row["last_token_logit_cosine"] for row in rows),
        "mean_last_token_absolute_difference": sum(
            row["last_token_mean_absolute_difference"] for row in rows
        )
        / count,
        "maximum_last_token_absolute_difference": max(
            row["last_token_max_absolute_difference"] for row in rows
        ),
        "mean_full_sequence_logit_cosine": sum(
            row["full_sequence_mean_logit_cosine"] for row in rows
        )
        / count,
        "minimum_full_sequence_logit_cosine": min(
            row["full_sequence_min_logit_cosine"] for row in rows
        ),
        "full_sequence_positions_compared": sum(
            row["full_sequence_positions_compared"] for row in rows
        ),
        "mean_full_sequence_absolute_difference": sum(
            row["full_sequence_mean_abs_logit_diff"] for row in rows
        )
        / count,
        "maximum_full_sequence_absolute_difference": max(
            row["full_sequence_max_abs_logit_diff"] for row in rows
        ),
        "top1_matches": sum(bool(row["top1_match"]) for row in rows),
        "output_exact_matches": sum(bool(row["output_exact_match"]) for row in rows),
        "mean_output_char_similarity": sum(row["output_char_similarity"] for row in rows) / count,
        "mean_output_token_sequence_similarity": sum(
            row["output_token_sequence_similarity"] for row in rows
        )
        / count,
        "mean_output_token_bag_cosine": sum(row["output_token_bag_cosine"] for row in rows)
        / count,
        "mcq_count": len(mcq_rows),
        "mcq_prediction_matches": sum(bool(row["mcq_prediction_match"]) for row in mcq_rows),
    }


def grouped_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("source", "task_category", "language_code"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row[field]), []).append(row)
        result[field] = {key: summarize_rows(value) for key, value in sorted(groups.items())}
    return result


def prompt_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
        )
    loss = float(output.loss.item())
    if not math.isfinite(loss):
        raise FloatingPointError("model produced a non-finite loss")
    if not bool(torch.isfinite(output.logits).all().item()):
        raise FloatingPointError("model produced non-finite logits")
    perplexity = math.exp(loss)
    last_index = int(attention_mask[0].sum().item()) - 1
    logits = output.logits[0, last_index].detach().float().cpu().contiguous()
    return perplexity, logits


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")
    manifest_rows = read_manifest(args.manifest)
    if args.smoke_limit is not None:
        if not 1 <= args.smoke_limit <= len(manifest_rows):
            raise SystemExit("--smoke-limit is outside the prompt manifest")
        manifest_rows = manifest_rows[: args.smoke_limit]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)

    tokenizer = _load_tokenizer(
        args.tokenizer,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    load_kwargs = {
        "model_loader": "hf-causal-lm",
        "dtype_name": args.dtype,
        "config_source": args.config,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    reference_model = load_model(args.reference, device, **load_kwargs)
    transformed_model = load_model(args.transformed, device, **load_kwargs)

    rows: list[dict[str, Any]] = []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    for manifest_row in manifest_rows:
        prompt = manifest_row["prompt"]
        encoded = _encode_prompt(tokenizer, prompt, device)
        reference_ppl, reference_last = prompt_forward(
            reference_model, encoded["input_ids"], encoded["attention_mask"]
        )
        transformed_ppl, transformed_last = prompt_forward(
            transformed_model, encoded["input_ids"], encoded["attention_mask"]
        )
        ppl_ratio = max(reference_ppl, transformed_ppl) / max(
            min(reference_ppl, transformed_ppl), 1e-12
        )
        last_difference = (reference_last - transformed_last).abs()
        # Cosine is mathematically bounded, but float32 roundoff can otherwise
        # produce values just above 1.0. Preserve the paper's metric while
        # enforcing its valid numeric range in the raw evidence.
        last_cosine = float(
            F.cosine_similarity(reference_last, transformed_last, dim=0)
            .clamp(-1.0, 1.0)
            .item()
        )

        reference_generated = _generate_ids(
            reference_model, tokenizer, prompt, args.max_new_tokens, device
        )
        transformed_generated = _generate_ids(
            transformed_model, tokenizer, prompt, args.max_new_tokens, device
        )
        reference_text = tokenizer.decode(reference_generated[0], skip_special_tokens=True)
        transformed_text = tokenizer.decode(transformed_generated[0], skip_special_tokens=True)
        output_metrics = _output_similarity_metrics(tokenizer, reference_text, transformed_text)
        full_attention_mask = torch.ones_like(reference_generated, device=device)
        full_metrics = _full_sequence_logit_metrics(
            reference_model, transformed_model, reference_generated, full_attention_mask
        )

        expected = manifest_row["expected"]
        if expected["kind"] == "multiple_choice":
            reference_mcq, _ = score_choices(
                reference_model, tokenizer, encoded["input_ids"], device
            )
            transformed_mcq, _ = score_choices(
                transformed_model, tokenizer, encoded["input_ids"], device
            )
            mcq_match: bool | None = reference_mcq == transformed_mcq
        else:
            reference_mcq = transformed_mcq = None
            mcq_match = None

        rows.append(
            {
                "prompt_id": manifest_row["prompt_id"],
                "source": manifest_row["source"],
                "task_category": manifest_row["task_category"],
                "language_code": manifest_row["language_code"],
                "expected_kind": expected["kind"],
                "input_token_count": int(encoded["input_ids"].shape[1]),
                "reference_perplexity": reference_ppl,
                "transformed_perplexity": transformed_ppl,
                "perplexity_ratio": ppl_ratio,
                "last_token_logit_cosine": last_cosine,
                "last_token_mean_absolute_difference": float(last_difference.mean().item()),
                "last_token_max_absolute_difference": float(last_difference.max().item()),
                "top1_match": int(reference_last.argmax().item())
                == int(transformed_last.argmax().item()),
                **full_metrics,
                **output_metrics,
                "reference_generated_token_ids": reference_generated[0].tolist(),
                "transformed_generated_token_ids": transformed_generated[0].tolist(),
                "reference_mcq_prediction": reference_mcq,
                "transformed_mcq_prediction": transformed_mcq,
                "mcq_prediction_match": mcq_match,
            }
        )

    require_complete_rows(rows, len(manifest_rows))
    aggregate = summarize_rows(rows)
    thresholds_passed = (
        aggregate["mean_last_token_logit_cosine"] >= args.min_cosine
        and aggregate["mean_perplexity_ratio"] <= args.max_ppl_ratio
    )
    git_commit = git_value("rev-parse", "HEAD")
    clean_checkout = git_value("status", "--porcelain") == ""
    reportable = (
        args.smoke_limit is None
        and len(rows) == 70
        and device.type == "cuda"
        and clean_checkout
        and thresholds_passed
    )
    payload = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_checkout_clean": clean_checkout,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "reference": args.reference,
        "transformed": args.transformed,
        "revision": args.revision,
        "tokenizer": _tokenizer_label(tokenizer),
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "config_fingerprint": config_fingerprint(reference_model.config),
        "dtype": args.dtype,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "command": [sys.executable, *sys.argv],
        "max_new_tokens": args.max_new_tokens,
        "thresholds": {
            "minimum_mean_last_token_logit_cosine": args.min_cosine,
            "maximum_mean_perplexity_ratio": args.max_ppl_ratio,
        },
        "thresholds_passed": thresholds_passed,
        "reported_eligible": reportable,
        "aggregate": aggregate,
        "by_manifest_dimension": grouped_summaries(rows),
        "prompts": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{'PASS' if thresholds_passed else 'FAIL'}: {len(rows)} prompts; "
        f"cosine={aggregate['mean_last_token_logit_cosine']:.8f}; "
        f"ppl_ratio={aggregate['mean_perplexity_ratio']:.8f}; "
        f"output={args.output}"
    )
    return 0 if thresholds_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
