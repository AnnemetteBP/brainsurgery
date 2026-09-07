#!/usr/bin/env python3
"""Run and preserve expanded versions of both paper behavioral comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml

from revision_tests.behavioral.run_cuda_matrix import (
    checkpoint_hashes,
    copy_model_sidecars,
    git_value,
    run,
    write_json,
)
from revision_tests.behavioral.run_paper_analysis import REQUIRED_PROMPT_METRICS
from revision_tests.scaling.oracle import compare_output, verify_huggingface_revision
from revision_tests.scaling.validate_protocol import EXPECTED_IDS, load_cases

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL_PATH = HERE / "paper_protocol.yaml"
PROTOCOL_ID = "eacl2027_behavioral_paper_v4"
COMPARISONS = ("imperative_equivalence", "regression_preservation")
REQUIRED_AGGREGATES = {
    "reference_mean_perplexity",
    "transformed_mean_perplexity",
    "mean_perplexity_ratio",
    "maximum_perplexity_ratio",
    "mean_last_token_logit_cosine",
    "minimum_last_token_logit_cosine",
    "mean_last_token_absolute_difference",
    "maximum_last_token_absolute_difference",
    "mean_full_sequence_logit_cosine",
    "minimum_full_sequence_logit_cosine",
    "mean_full_sequence_absolute_difference",
    "maximum_full_sequence_absolute_difference",
    "top1_matches",
    "output_exact_matches",
    "mean_output_char_similarity",
    "mean_output_token_sequence_similarity",
    "mean_output_token_bag_cosine",
    "mcq_prediction_matches",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=f"eacl2027_behavioral_paper_cuda_{git_value('rev-parse', '--short', 'HEAD')}",
    )
    parser.add_argument("--model", action="append", choices=EXPECTED_IDS, dest="models")
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--keep-transformed", action="store_true")
    return parser.parse_args()


def load_protocol() -> dict[str, Any]:
    value = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"paper_protocol.yaml must declare {PROTOCOL_ID}")
    if value.get("expected_model_ids") != EXPECTED_IDS:
        raise ValueError("the complete model matrix changed")
    required = set(value.get("required_old_paper_metrics", []))
    if required != REQUIRED_PROMPT_METRICS - {"mcq_prediction_match"}:
        raise ValueError("the original paper metric set changed")
    operation = value.get("operation", {})
    if (operation.get("target_regex"), operation.get("forward_factor"), operation.get("backward_factor")) != (r".*\.weight", 0.5, 2.0):
        raise ValueError("the forward and forward-backward operations changed")
    reporting = value.get("reporting", {})
    if reporting.get("required_comparisons") != list(COMPARISONS):
        raise ValueError("both original paper comparisons are required")
    if reporting.get("restored_tensor_reference") != "independent_pytorch_forward_backward":
        raise ValueError("the restored tensor oracle must use the independent PyTorch round trip")
    if not all(reporting.get(key) is True for key in ("preserve_per_prompt_metrics", "preserve_per_model_aggregates", "forbid_placeholder_tables", "forbid_nonfinite_metrics")):
        raise ValueError("evidence preservation gates must remain enabled")
    return value


def write_plan(path: Path, source: Path, output: Path, protocol: dict[str, Any], *, restore: bool) -> None:
    operation = protocol["operation"]
    transforms: list[dict[str, Any]] = [
        {"scale_": {"target": operation["target_regex"], "by": operation["forward_factor"]}}
    ]
    if restore:
        namespace = operation["temporary_namespace"]
        transforms.extend(
            [
                {"scale_": {"target": operation["target_regex"], "by": operation["backward_factor"]}},
                {"move": {"from": r"^(.*)$", "to": rf"{namespace}.\1"}},
                {"move": {"from": rf"^{namespace}\.(.*)$", "to": r"\1"}},
            ]
        )
    value = {
        "inputs": [f"model::{source}"],
        "transforms": transforms,
        "output": {"path": str(output), "format": "safetensors", "shard": operation["output_shard_size"]},
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def validate_result(result: dict[str, Any], expected_prompts: int) -> None:
    if result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("wrong behavioral protocol")
    prompts = result.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != expected_prompts:
        raise ValueError("per-prompt evidence is incomplete")
    for row in prompts:
        missing = REQUIRED_PROMPT_METRICS - row.keys()
        if missing:
            raise ValueError(f"{row.get('prompt_id')}: missing {sorted(missing)}")
    missing = REQUIRED_AGGREGATES - result.get("aggregate", {}).keys()
    if missing:
        raise ValueError(f"aggregate evidence is incomplete: {sorted(missing)}")
    if set(result.get("by_manifest_dimension", {})) != {"source", "task_category", "language_code"}:
        raise ValueError("source/task/language evidence is incomplete")
    if not isinstance(result.get("reported_eligible"), bool):
        raise ValueError("reported eligibility is missing")


def run_pair(reference: Path, transformed: Path, source: Path, case: dict[str, Any], protocol: dict[str, Any], output: Path, smoke_limit: int | None) -> dict[str, Any]:
    command = [
        str(REPO / ".venv/bin/python"), str(HERE / "run_paper_analysis.py"),
        "--reference", str(reference), "--transformed", str(transformed),
        "--config", str(source), "--tokenizer", str(source), "--revision", case["revision"],
        "--device", "cuda:0", "--dtype", case["expected_weight_dtype"],
        "--max-new-tokens", str(protocol["inference"]["max_new_tokens"]),
        "--min-cosine", str(protocol["thresholds"]["minimum_mean_last_token_logit_cosine"]),
        "--max-ppl-ratio", str(protocol["thresholds"]["maximum_mean_perplexity_ratio"]),
        "--local-files-only", "--output", str(output),
    ]
    if smoke_limit is not None:
        command.extend(["--smoke-limit", str(smoke_limit)])
    run(command)
    result = json.loads(output.read_text(encoding="utf-8"))
    validate_result(result, smoke_limit or 70)
    retained = (
        "manifest_sha256",
        "revision",
        "tokenizer_fingerprint",
        "config_fingerprint",
        "dtype",
        "device",
        "gpu",
        "max_new_tokens",
        "thresholds",
        "thresholds_passed",
        "reported_eligible",
        "aggregate",
        "by_manifest_dimension",
        "prompts",
    )
    return {key: result[key] for key in retained}


def render_markdown(evidence: dict[str, Any]) -> str:
    lines = [
        "# Expanded behavioral analysis", "",
        f"Protocol: `{evidence['protocol_id']}`  ", f"Run: `{evidence['run_id']}`  ",
        f"Commit: `{evidence['git_commit']}`  ", f"GPU: {evidence['gpu']}", "",
        "| Model | BS/PyTorch PPL ratio | BS/PyTorch cosine | BS/PyTorch top-1 | Original/restored sequence cosine | Original/restored exact output |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in evidence["results"]:
        eq = case["comparisons"]["imperative_equivalence"]["aggregate"]
        reg = case["comparisons"]["regression_preservation"]["aggregate"]
        n = eq["prompt_count"]
        lines.append(f"| {case['display']} | {eq['mean_perplexity_ratio']:.8f} | {eq['mean_last_token_logit_cosine']:.8f} | {eq['top1_matches']}/{n} | {reg['mean_full_sequence_logit_cosine']:.8f} | {reg['output_exact_matches']}/{n} |")
    lines.extend(["", "Every original metric and every per-prompt value is retained in `evidence.json`.", ""])
    return "\n".join(lines)


def render_latex(evidence: dict[str, Any]) -> str:
    rows = []
    for case in evidence["results"]:
        eq = case["comparisons"]["imperative_equivalence"]["aggregate"]
        reg = case["comparisons"]["regression_preservation"]["aggregate"]
        n = eq["prompt_count"]
        rows.append(f"{case['display']} & {eq['mean_perplexity_ratio']:.8f} & {eq['mean_last_token_logit_cosine']:.8f} & {eq['top1_matches']}/{n} & {reg['mean_full_sequence_logit_cosine']:.8f} & {reg['output_exact_matches']}/{n} \\\\ ")
    return "\n".join([
        r"\begin{table*}[t]", r"\centering", r"\small", r"\begin{tabular}{lrrrrr}",
        r"\toprule", r"Model & PPL ratio & Final cosine & Top-1 & Sequence cosine & Exact output \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Expanded behavioral analysis on 70 sourced prompts per checkpoint. PPL ratio, final-token cosine, and top-1 compare BrainSurgery with an independent imperative implementation; sequence cosine and exact output compare the original checkpoint with its forward--backward restored checkpoint.}",
        r"\label{tab:expanded-behavioral}", r"\end{table*}", "",
    ])


def paper_totals(evidence: dict[str, Any]) -> dict[str, Any]:
    equivalence = [
        result["comparisons"]["imperative_equivalence"]["aggregate"]
        for result in evidence["results"]
    ]
    regression = [
        result["comparisons"]["regression_preservation"]["aggregate"]
        for result in evidence["results"]
    ]
    return {
        "models": len(evidence["results"]),
        "prompts": sum(value["prompt_count"] for value in equivalence),
        "equivalence_ppl_ratio_min": min(value["mean_perplexity_ratio"] for value in equivalence),
        "equivalence_ppl_ratio_max": max(value["mean_perplexity_ratio"] for value in equivalence),
        "equivalence_final_cosine_min": min(value["minimum_last_token_logit_cosine"] for value in equivalence),
        "equivalence_last_abs_diff_max": max(value["maximum_last_token_absolute_difference"] for value in equivalence),
        "equivalence_top1_matches": sum(value["top1_matches"] for value in equivalence),
        "regression_sequence_cosine_min": min(value["minimum_full_sequence_logit_cosine"] for value in regression),
        "regression_sequence_abs_diff_max": max(value["maximum_full_sequence_absolute_difference"] for value in regression),
        "regression_positions": sum(value["full_sequence_positions_compared"] for value in regression),
        "regression_exact_outputs": sum(value["output_exact_matches"] for value in regression),
        "regression_mcq_matches": sum(value["mcq_prediction_matches"] for value in regression),
        "regression_mcq_count": sum(value["mcq_count"] for value in regression),
        "scaled_tensors": sum(
            result["tensor_oracle"]["brainsurgery_scaled"]["tensors_checked"]
            for result in evidence["results"]
        ),
        "restored_tensors": sum(
            result["tensor_oracle"]["brainsurgery_restored"]["tensors_checked"]
            for result in evidence["results"]
        ),
    }


def render_paper_text(evidence: dict[str, Any], *, latex: bool = False) -> str:
    values = paper_totals(evidence)
    prefix = "\\paragraph{Behavioral equivalence and preservation.}\n" if latex else "# Behavioral equivalence and preservation\n\n"
    models = values["models"]
    prompts = values["prompts"]
    text = (
        f"Across {models} pinned checkpoints and {prompts} sourced prompt--model pairs, "
        "we retained the two behavioral comparisons from the previous evaluation. "
        "An independent tensor oracle first matched all "
        f"{values['scaled_tensors']} tensors in the BrainSurgery scaling outputs to "
        "the direct-PyTorch outputs and all "
        f"{values['restored_tensors']} tensors in the BrainSurgery forward--backward "
        "outputs to independent direct-PyTorch forward--backward outputs. "
        "For the meaningful weight-scaling operation, BrainSurgery and a separate "
        "direct-PyTorch implementation had per-model mean perplexity ratios in "
        f"[{values['equivalence_ppl_ratio_min']:.8f}, {values['equivalence_ppl_ratio_max']:.8f}], "
        f"minimum final-token logit cosine {values['equivalence_final_cosine_min']:.8f}, "
        f"maximum absolute final-token logit difference {values['equivalence_last_abs_diff_max']:.8g}, "
        f"and {values['equivalence_top1_matches']}/{prompts} top-1 agreement. "
        "For the forward--backward transform, the restored and original checkpoints "
        f"had minimum per-position full-vocabulary logit cosine {values['regression_sequence_cosine_min']:.8f} "
        f"over {values['regression_positions']} aligned positions, maximum absolute logit "
        f"difference {values['regression_sequence_abs_diff_max']:.8g}, "
        f"{values['regression_exact_outputs']}/{prompts} exact greedy outputs, and "
        f"{values['regression_mcq_matches']}/{values['regression_mcq_count']} matching "
        "multiple-choice predictions. These measurements establish equivalence only "
        "for the enumerated transformations, checkpoints, prompts, dtypes, and runtime."
    )
    if latex:
        text = text.replace("direct-PyTorch", "direct-PyTorch").replace("--", r"--")
    return prefix + text + "\n"


def run_case(args: argparse.Namespace, case: dict[str, Any], protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    source = REPO / case["input"]
    if not source.exists() or not verify_huggingface_revision(source, case["revision"])["passed"]:
        raise RuntimeError(f"{case['id']}: source or pinned revision invalid")
    case_root = run_root / case["id"].lower()
    generated = REPO / "models" / "behavioral_paper_v4" / case["id"].lower()
    python_scaled = generated / "python_scaled"
    python_restored = generated / "python_restored"
    bs_scaled = generated / "brainsurgery_scaled"
    restored = generated / "brainsurgery_restored"
    if case_root.exists() or generated.exists():
        raise RuntimeError(f"{case['id']}: refusing to overwrite prior artifacts")
    case_root.mkdir(parents=True)
    source_before = checkpoint_hashes(source)
    scaled_plan, restored_plan = case_root / "scaled.yaml", case_root / "restored.yaml"
    write_plan(scaled_plan, source, bs_scaled, protocol, restore=False)
    write_plan(restored_plan, source, restored, protocol, restore=True)
    common = ["--provider", "inmemory", "--num-workers", "1", "--no-summarize", "--log-level", "warning"]
    run([str(REPO / ".venv/bin/brainsurgery"), str(scaled_plan), *common])
    run([str(REPO / ".venv/bin/brainsurgery"), str(restored_plan), *common])
    operation = protocol["operation"]
    run([str(REPO / ".venv/bin/python"), str(REPO / "revision_tests/scaling/baseline.py"), "--input", str(source), "--output", str(python_scaled), "--target-regex", operation["target_regex"], "--factor", str(operation["forward_factor"]), "--shard-size-bytes", str(operation["output_shard_size_bytes"])])
    run([str(REPO / ".venv/bin/python"), str(REPO / "revision_tests/scaling/baseline.py"), "--input", str(python_scaled), "--output", str(python_restored), "--target-regex", operation["target_regex"], "--factor", str(operation["backward_factor"]), "--shard-size-bytes", str(operation["output_shard_size_bytes"])])
    for output in (python_scaled, python_restored, bs_scaled, restored):
        copy_model_sidecars(source, output)
    oracle_specs = {
        "python_scaled": (python_scaled, operation["target_regex"], operation["forward_factor"]),
        "brainsurgery_scaled": (bs_scaled, operation["target_regex"], operation["forward_factor"]),
        "brainsurgery_restored": (restored, operation["target_regex"], 1.0),
    }
    oracles = {
        name: compare_output(
            python_restored if name == "brainsurgery_restored" else source,
            output,
            target_regex=regex,
            factor=factor,
            shard_size_bytes=operation["output_shard_size_bytes"],
        )
        for name, (output, regex, factor) in oracle_specs.items()
    }
    if not all(value["passed"] for value in oracles.values()) or checkpoint_hashes(source) != source_before:
        raise RuntimeError(f"{case['id']}: independent tensor oracle failed")
    comparisons = {
        "imperative_equivalence": run_pair(python_scaled, bs_scaled, source, case, protocol, case_root / "imperative_equivalence.json", args.smoke_limit),
        "regression_preservation": run_pair(source, restored, source, case, protocol, case_root / "regression_preservation.json", args.smoke_limit),
    }
    result = {
        key: case[key] for key in ("id", "display", "family", "nominal_parameter_count", "model_id", "revision")
    }
    tensor_oracle = {
        name: {
            **{key: value[key] for key in ("passed", "tensors_checked", "tensors_passed")},
            "reference": (
                "independent_pytorch_forward_backward"
                if name == "brainsurgery_restored"
                else "original_checkpoint"
            ),
        }
        for name, value in oracles.items()
    }
    result.update({"dtype": case["expected_weight_dtype"], "tensor_oracle": tensor_oracle, "comparisons": comparisons})
    if not args.keep_transformed:
        shutil.rmtree(generated)
    return result


def main() -> int:
    args = parse_args()
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise SystemExit("the reported matrix requires Linux and CUDA")
    if git_value("status", "--porcelain"):
        raise SystemExit("the Git checkout must be clean")
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 70:
        raise SystemExit("--smoke-limit must be between 1 and 70")
    protocol, cases_doc = load_protocol(), load_cases()
    selected = args.models or EXPECTED_IDS
    cases = [case for case in cases_doc["models"] if case["id"] in selected]
    run_root, evidence_root = REPO / "log" / "revision_tests" / args.run_id / "behavioral_paper", HERE / "results" / args.run_id
    if run_root.exists() or evidence_root.exists():
        raise SystemExit("refusing to overwrite an existing run")
    run([str(REPO / ".venv/bin/python"), str(HERE / "validate_manifest.py")])
    run([str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", str(HERE / "test_paper_analysis.py"), str(HERE / "test_cuda_paper_matrix.py")])
    results = [run_case(args, case, protocol, run_root) for case in cases]
    reportable = (
        args.smoke_limit is None
        and selected == EXPECTED_IDS
        and len(results) == 10
        and all(
            all(
                comparison["thresholds_passed"]
                and comparison["reported_eligible"]
                for comparison in result["comparisons"].values()
            )
            for result in results
        )
    )
    evidence = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "run_id": args.run_id,
        "git_commit": git_value("rev-parse", "HEAD"),
        # Keep the committed evidence anonymous and reproducible: do not
        # preserve machine-specific absolute executable or checkout paths.
        "command": [
            ".venv/bin/python",
            "revision_tests/behavioral/run_cuda_paper_matrix.py",
            "--run-id",
            args.run_id,
        ],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "operation": protocol["operation"],
        "reported_eligible": reportable,
        "raw_summary_location": f"log/revision_tests/{args.run_id}/behavioral_paper/summary.json",
        "results": results,
    }
    write_json(run_root / "summary.json", evidence)
    if not reportable:
        print("NON-REPORTABLE: no paper artifacts generated", flush=True)
        return 0
    evidence_root.mkdir(parents=True)
    write_json(evidence_root / "evidence.json", evidence)
    (evidence_root / "table.md").write_text(render_markdown(evidence), encoding="utf-8")
    (evidence_root / "table.tex").write_text(render_latex(evidence), encoding="utf-8")
    (evidence_root / "paper_text.md").write_text(
        render_paper_text(evidence), encoding="utf-8"
    )
    (evidence_root / "paper_text.tex").write_text(
        render_paper_text(evidence, latex=True), encoding="utf-8"
    )
    print(f"REPORTABLE: complete evidence preserved at {evidence_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
