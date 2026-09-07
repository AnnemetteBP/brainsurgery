#!/usr/bin/env python3
"""Run structural losslessness and the Pythia dtype/precision ablation on CUDA."""

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
from revision_tests.behavioral.run_cuda_paper_matrix import run_pair
from revision_tests.scaling.oracle import compare_output, verify_huggingface_revision
from revision_tests.scaling.validate_protocol import EXPECTED_IDS, load_cases

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL_PATH = HERE / "methodology_protocol.yaml"
PROTOCOL_ID = "eacl2027_behavioral_methodology_v5"
SHARD_BYTES = 268435456


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=f"eacl2027_behavioral_methodology_cuda_{git_value('rev-parse', '--short', 'HEAD')}",
    )
    parser.add_argument("--model", action="append", choices=EXPECTED_IDS, dest="models")
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--keep-transformed", action="store_true")
    return parser.parse_args()


def load_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"methodology_protocol.yaml must declare {PROTOCOL_ID}")
    if protocol["expected_model_ids"] != EXPECTED_IDS:
        raise ValueError("the ten-model matrix changed")
    if protocol["structural_losslessness"]["arithmetic_allowed"] is not False:
        raise ValueError("structural losslessness must forbid arithmetic")
    if protocol["pythia_precision_ablation"]["model_ids"] != EXPECTED_IDS[:4]:
        raise ValueError("the precision ablation must contain exactly the four Pythia models")
    return protocol


def write_bs_plan(source: Path, output: Path, transforms: list[dict[str, Any]], path: Path) -> None:
    payload = {
        "inputs": [f"model::{source}"],
        "transforms": transforms,
        "output": {"path": str(output), "format": "safetensors", "shard": "256MB"},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def exact_oracle(reference: Path, output: Path) -> dict[str, Any]:
    result = compare_output(
        reference,
        output,
        target_regex=r".*\.weight",
        factor=1.0,
        shard_size_bytes=SHARD_BYTES,
    )
    return {
        "passed": result["passed"],
        "tensors_checked": result["tensors_checked"],
        "tensors_passed": result["tensors_passed"],
        "missing_tensors": result["missing_tensors"],
        "unexpected_tensors": result["unexpected_tensors"],
    }


def annotate_health(comparison: dict[str, Any]) -> dict[str, Any]:
    numeric_fields = (
        "reference_perplexity", "transformed_perplexity", "perplexity_ratio",
        "last_token_logit_cosine", "last_token_mean_absolute_difference",
        "last_token_max_absolute_difference", "full_sequence_mean_logit_cosine",
        "full_sequence_min_logit_cosine", "full_sequence_mean_abs_logit_diff",
        "full_sequence_max_abs_logit_diff", "output_char_similarity",
        "output_token_sequence_similarity", "output_token_bag_cosine",
    )
    nan_count = 0
    inf_count = 0
    for row in comparison["prompts"]:
        for field in numeric_fields:
            value = float(row[field])
            nan_count += int(torch.isnan(torch.tensor(value)).item())
            inf_count += int(torch.isinf(torch.tensor(value)).item())
    comparison["numeric_health"] = {
        "values_checked": len(comparison["prompts"]) * len(numeric_fields),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "passed": nan_count == 0 and inf_count == 0,
    }
    return comparison


def analyze_pair(
    reference: Path,
    transformed: Path,
    source: Path,
    case: dict[str, Any],
    protocol: dict[str, Any],
    output: Path,
    smoke_limit: int | None,
) -> dict[str, Any]:
    result = annotate_health(
        run_pair(reference, transformed, source, case, protocol, output, smoke_limit)
    )
    result["command"] = json.loads(output.read_text(encoding="utf-8"))["command"]
    return result


def structural_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    protocol: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    source = REPO / case["input"]
    if not verify_huggingface_revision(source, case["revision"])["passed"]:
        raise RuntimeError(f"{case['id']}: pinned source revision failed")
    source_before = checkpoint_hashes(source)
    case_root = run_root / "structural_losslessness" / case["id"].lower()
    generated = REPO / "models" / args.run_id / "structural" / case["id"].lower()
    pytorch_restored = generated / "pytorch_restored"
    bs_restored = generated / "brainsurgery_restored"
    case_root.mkdir(parents=True)
    namespace = protocol["structural_losslessness"]["temporary_namespace"]
    plan = case_root / "brainsurgery_plan.yaml"
    transforms = [
        {"move": {"from": r"^(.*)$", "to": rf"{namespace}.\1"}},
        {"move": {"from": rf"^{namespace}\.(.*)$", "to": r"\1"}},
    ]
    write_bs_plan(source, bs_restored, transforms, plan)
    bs_command = [
        str(REPO / ".venv/bin/brainsurgery"), str(plan), "--provider", "inmemory",
        "--num-workers", "1", "--no-summarize", "--log-level", "warning",
    ]
    py_command = [
        str(REPO / ".venv/bin/python"), str(HERE / "methodology_baseline.py"),
        "--input", str(source), "--output", str(pytorch_restored),
        "--operation", "structural-roundtrip", "--temporary-namespace", namespace,
        "--shard-size-bytes", str(SHARD_BYTES),
    ]
    run(bs_command)
    run(py_command)
    for output in (pytorch_restored, bs_restored):
        copy_model_sidecars(source, output)
    oracles = {
        "original_vs_brainsurgery": exact_oracle(source, bs_restored),
        "original_vs_pytorch": exact_oracle(source, pytorch_restored),
        "pytorch_vs_brainsurgery": exact_oracle(pytorch_restored, bs_restored),
    }
    if not all(item["passed"] for item in oracles.values()):
        raise RuntimeError(f"{case['id']}: structural tensor oracle failed")
    comparisons = {
        "original_vs_brainsurgery": analyze_pair(
            source, bs_restored, source, case, protocol,
            case_root / "original_vs_brainsurgery.json", args.smoke_limit,
        ),
        "pytorch_vs_brainsurgery": analyze_pair(
            pytorch_restored, bs_restored, source, case, protocol,
            case_root / "pytorch_vs_brainsurgery.json", args.smoke_limit,
        ),
    }
    if checkpoint_hashes(source) != source_before:
        raise RuntimeError(f"{case['id']}: source checkpoint changed")
    if not args.keep_transformed:
        shutil.rmtree(generated)
    return {
        **{key: case[key] for key in ("id", "display", "family", "nominal_parameter_count", "model_id", "revision")},
        "dtype": case["expected_weight_dtype"],
        "operation": "namespace_move_restore_without_arithmetic",
        "commands": {"brainsurgery": bs_command, "independent_pytorch": py_command},
        "raw_results": {
            name: str((case_root / f"{name}.json").relative_to(REPO))
            for name in comparisons
        },
        "tensor_oracles": oracles,
        "comparisons": comparisons,
    }


def cast_fp32(source: Path, output: Path) -> list[str]:
    command = [
        str(REPO / ".venv/bin/python"), str(HERE / "methodology_baseline.py"),
        "--input", str(source), "--output", str(output), "--operation", "cast-float32",
        "--shard-size-bytes", str(SHARD_BYTES),
    ]
    run(command)
    copy_model_sidecars(source, output)
    return command


def ablation_arm(
    args: argparse.Namespace,
    case: dict[str, Any],
    protocol: dict[str, Any],
    run_root: Path,
    arm: dict[str, Any],
) -> dict[str, Any]:
    original_source = REPO / case["input"]
    source_before = checkpoint_hashes(original_source)
    arm_id = arm["id"]
    case_root = run_root / "pythia_precision_ablation" / case["id"].lower() / arm_id
    generated = REPO / "models" / args.run_id / "ablation" / case["id"].lower() / arm_id
    source = original_source
    commands: dict[str, Any] = {}
    if arm_id == "fp32_throughout":
        source = generated / "fp32_original"
        commands["cast_to_fp32"] = cast_fp32(original_source, source)
    pytorch_scaled = generated / "pytorch_scaled"
    pytorch_restored = generated / "pytorch_restored"
    bs_restored = generated / "brainsurgery_restored"
    case_root.mkdir(parents=True)
    operation = protocol["pythia_precision_ablation"]["operation"]
    plan = case_root / "brainsurgery_plan.yaml"
    transforms = [
        {"scale_": {"target": operation["target_regex"], "by": operation["forward_factor"]}},
        {"scale_": {"target": operation["target_regex"], "by": operation["backward_factor"]}},
    ]
    write_bs_plan(source, bs_restored, transforms, plan)
    bs_command = [
        str(REPO / ".venv/bin/brainsurgery"), str(plan), "--provider", "inmemory",
        "--num-workers", "1", "--no-summarize", "--log-level", "warning",
    ]
    commands["brainsurgery"] = bs_command
    run(bs_command)
    baseline = REPO / "revision_tests/scaling/baseline.py"
    py_scale = [
        str(REPO / ".venv/bin/python"), str(baseline), "--input", str(source),
        "--output", str(pytorch_scaled), "--target-regex", operation["target_regex"],
        "--factor", str(operation["forward_factor"]), "--shard-size-bytes", str(SHARD_BYTES),
    ]
    py_restore = [
        str(REPO / ".venv/bin/python"), str(baseline), "--input", str(pytorch_scaled),
        "--output", str(pytorch_restored), "--target-regex", operation["target_regex"],
        "--factor", str(operation["backward_factor"]), "--shard-size-bytes", str(SHARD_BYTES),
    ]
    commands["independent_pytorch"] = [py_scale, py_restore]
    run(py_scale)
    run(py_restore)
    for output in (pytorch_scaled, pytorch_restored, bs_restored):
        copy_model_sidecars(source, output)
    oracles = {
        "original_vs_brainsurgery": exact_oracle(source, bs_restored),
        "original_vs_pytorch": exact_oracle(source, pytorch_restored),
        "pytorch_vs_brainsurgery": exact_oracle(pytorch_restored, bs_restored),
    }
    analysis_case = dict(case, expected_weight_dtype=arm["inference_dtype"])
    comparisons = {
        "original_vs_brainsurgery": analyze_pair(
            source, bs_restored, source, analysis_case, protocol,
            case_root / "original_vs_brainsurgery.json", args.smoke_limit,
        ),
        "pytorch_vs_brainsurgery": analyze_pair(
            pytorch_restored, bs_restored, source, analysis_case, protocol,
            case_root / "pytorch_vs_brainsurgery.json", args.smoke_limit,
        ),
    }
    if checkpoint_hashes(original_source) != source_before:
        raise RuntimeError(f"{case['id']}/{arm_id}: source checkpoint changed")
    if not args.keep_transformed:
        shutil.rmtree(generated)
    return {
        **{key: case[key] for key in ("id", "display", "model_id", "revision")},
        "arm": arm_id,
        "storage_dtype": arm["storage_dtype"],
        "inference_dtype": arm["inference_dtype"],
        "cast_back_to_float16": arm.get("cast_back_to_float16", False),
        "operation": "multiply_0.5_save_multiply_2",
        "commands": commands,
        "raw_results": {
            name: str((case_root / f"{name}.json").relative_to(REPO))
            for name in comparisons
        },
        "tensor_oracles": oracles,
        "comparisons": comparisons,
    }


def collect_failures(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    failures = []
    for item in results:
        label = item["id"] + ("/" + item["arm"] if "arm" in item else "")
        for name, oracle in item["tensor_oracles"].items():
            if not oracle["passed"]:
                failures.append({"case": label, "check": name, "reason": "tensor oracle failed"})
        for name, comparison in item["comparisons"].items():
            if not comparison["thresholds_passed"]:
                failures.append({"case": label, "check": name, "reason": "behavioral threshold failed"})
            if not comparison["numeric_health"]["passed"]:
                failures.append({"case": label, "check": name, "reason": "NaN or Inf detected"})
    return failures


def metric_row(item: dict[str, Any], comparison: str) -> tuple[Any, ...]:
    aggregate = item["comparisons"][comparison]["aggregate"]
    health = item["comparisons"][comparison]["numeric_health"]
    oracle = item["tensor_oracles"]["original_vs_brainsurgery"]
    return (
        aggregate["prompt_count"], oracle["tensors_passed"], oracle["tensors_checked"],
        aggregate["mean_perplexity_ratio"], aggregate["mean_last_token_logit_cosine"],
        aggregate["mean_full_sequence_logit_cosine"],
        aggregate["maximum_full_sequence_absolute_difference"],
        aggregate["output_exact_matches"], aggregate["mcq_prediction_matches"],
        aggregate["mcq_count"], health["nan_count"], health["inf_count"],
    )


def render_markdown(evidence: dict[str, Any]) -> str:
    lines = [
        "# Behavioral methodology correction", "",
        "## Structural losslessness", "",
        "This is the primary losslessness evidence. It uses only a namespace move and exact name restoration; no arithmetic or dtype conversion is performed.", "",
        "| **Model** | **Dtype** | **Prompts** | **Exact tensors ↑** | **PPL ratio ↓** | **Final cosine ↑** | **Sequence cosine ↑** | **Max abs diff ↓** | **Exact generations ↑** | **MCQ agreement ↑** | **NaN** | **Inf** |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in evidence["structural_losslessness"]["results"]:
        n, tp, tc, ppl, final, seq, diff, exact, mcq, mcqn, nan, inf = metric_row(item, "original_vs_brainsurgery")
        lines.append(f"| {item['display']} | {item['dtype']} | {n} | {tp}/{tc} | {ppl:.8f} | {final:.8f} | {seq:.8f} | {diff:.8g} | {exact}/{n} | {mcq}/{mcqn} | {nan} | {inf} |")
    lines += ["", "## Pythia precision ablation", "", "This is a dtype/precision ablation, not the primary losslessness test. The old native-FP16 arithmetic round trip is not used as primary losslessness evidence.", "",
        "| **Model** | **Storage/inference dtype** | **Prompts** | **Exact tensors ↑** | **PPL ratio ↓** | **Final cosine ↑** | **Sequence cosine ↑** | **Max abs diff ↓** | **Exact generations ↑** | **MCQ agreement ↑** | **NaN** | **Inf** |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in evidence["pythia_precision_ablation"]["results"]:
        n, tp, tc, ppl, final, seq, diff, exact, mcq, mcqn, nan, inf = metric_row(item, "original_vs_brainsurgery")
        dtype = f"{item['storage_dtype']}/{item['inference_dtype']}"
        lines.append(f"| {item['display']} ({item['arm']}) | {dtype} | {n} | {tp}/{tc} | {ppl:.8f} | {final:.8f} | {seq:.8f} | {diff:.8g} | {exact}/{n} | {mcq}/{mcqn} | {nan} | {inf} |")
    return "\n".join(lines) + "\n"


def render_latex(evidence: dict[str, Any]) -> str:
    def section(title: str, rows: list[dict[str, Any]], ablation: bool) -> list[str]:
        output = [
            rf"\multicolumn{{12}}{{l}}{{\textbf{{{title}}}}} \\",
            r"\textbf{Model} & \textbf{Dtype} & \textbf{N} & \textbf{Exact tensors $\uparrow$} & \textbf{PPL $\downarrow$} & \textbf{Final cos. $\uparrow$} & \textbf{Seq. cos. $\uparrow$} & \textbf{Max diff $\downarrow$} & \textbf{Exact gen. $\uparrow$} & \textbf{MCQ $\uparrow$} & \textbf{NaN} & \textbf{Inf} \\",
            r"\midrule",
        ]
        for item in rows:
            n, tp, tc, ppl, final, seq, diff, exact, mcq, mcqn, nan, inf = metric_row(item, "original_vs_brainsurgery")
            dtype = item["dtype"] if not ablation else item["storage_dtype"]
            label = item["display"] if not ablation else f"{item['display']} ({item['arm']})"
            output.append(f"{label} & {dtype} & {n} & {tp}/{tc} & {ppl:.8f} & {final:.8f} & {seq:.8f} & {diff:.8g} & {exact}/{n} & {mcq}/{mcqn} & {nan} & {inf} \\\\")
        return output
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\begin{tabular}{llrrrrrrrrrr}", r"\toprule"]
    lines += section("Structural losslessness", evidence["structural_losslessness"]["results"], False)
    lines += [r"\midrule"]
    lines += section("Pythia precision ablation", evidence["pythia_precision_ablation"]["results"], True)
    lines += [r"\bottomrule", r"\end{tabular}", r"\caption{Structural losslessness and the separate Pythia precision ablation. The native-FP16 arithmetic round trip is not the primary losslessness test.}", r"\label{tab:behavioral-methodology}", r"\end{table*}", ""]
    return "\n".join(lines)


def render_paper_text(evidence: dict[str, Any], latex: bool = False) -> str:
    structural = evidence["structural_losslessness"]["results"]
    ablation = evidence["pythia_precision_ablation"]["results"]
    structural_prompts = sum(item["comparisons"]["original_vs_brainsurgery"]["aggregate"]["prompt_count"] for item in structural)
    ablation_prompts = sum(item["comparisons"]["original_vs_brainsurgery"]["aggregate"]["prompt_count"] for item in ablation)
    heading1 = r"\paragraph{Structural losslessness.}" if latex else "## Structural losslessness"
    heading2 = r"\paragraph{Pythia precision ablation.}" if latex else "## Pythia precision ablation"
    return (
        f"{heading1}\n"
        f"The primary losslessness evaluation applied a value-preserving namespace move and restoration, without arithmetic or dtype conversion, to {len(structural)} checkpoints at native dtype and evaluated {structural_prompts} prompt--model pairs. Exact tensor, behavioral, and non-finite checks are reported in Table~\\ref{{tab:behavioral-methodology}}.\n\n"
        f"{heading2}\n"
        f"The secondary dtype/precision ablation evaluated {len(ablation)} Pythia model/dtype arms and {ablation_prompts} prompt--model pairs under the multiply-by-0.5 then multiply-by-2 save round trip. FP32 storage and inference were retained throughout the FP32 arm and were never cast back to FP16. The old native-FP16 arithmetic round trip is not the primary losslessness test. All negative results are retained.\n"
    )


def validate_counts(evidence: dict[str, Any], expected_models: int, smoke_limit: int | None) -> None:
    prompts = smoke_limit or 70
    structural = evidence["structural_losslessness"]["results"]
    ablation = evidence["pythia_precision_ablation"]["results"]
    if len(structural) != expected_models:
        raise ValueError("structural model count is incomplete")
    if expected_models == 10 and len(ablation) != 8:
        raise ValueError("ablation model/dtype count is incomplete")
    for section in (structural, ablation):
        for item in section:
            for comparison in item["comparisons"].values():
                if len(comparison["prompts"]) != prompts or comparison["aggregate"]["prompt_count"] != prompts:
                    raise ValueError("raw prompt and aggregate counts disagree")
                if comparison["numeric_health"]["nan_count"] or comparison["numeric_health"]["inf_count"]:
                    raise ValueError("non-finite metric found")


def main() -> int:
    args = parse_args()
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise SystemExit("the methodology matrix requires Linux and CUDA")
    if git_value("status", "--porcelain"):
        raise SystemExit("the Git checkout must be clean")
    protocol, cases_doc = load_protocol(), load_cases()
    selected = args.models or EXPECTED_IDS
    cases = [case for case in cases_doc["models"] if case["id"] in selected]
    run_root = REPO / "log" / "revision_tests" / args.run_id / "behavioral_methodology"
    evidence_root = HERE / "results" / args.run_id
    if run_root.exists() or evidence_root.exists() or (REPO / "models" / args.run_id).exists():
        raise SystemExit("refusing to overwrite an existing methodology run")
    run([str(REPO / ".venv/bin/python"), str(HERE / "validate_manifest.py")])
    structural = [structural_case(args, case, protocol, run_root) for case in cases]
    ablation = []
    for case in cases:
        if case["id"] in protocol["pythia_precision_ablation"]["model_ids"]:
            for arm in protocol["pythia_precision_ablation"]["arms"]:
                ablation.append(ablation_arm(args, case, protocol, run_root, arm))
    evidence = {
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "run_id": args.run_id,
        "git_commit": git_value("rev-parse", "HEAD"),
        "command": [".venv/bin/python", "revision_tests/behavioral/run_cuda_methodology.py", "--run-id", args.run_id],
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "raw_summary_location": str((run_root / "summary.json").relative_to(REPO)),
        "structural_losslessness": {
            "role": "primary_paper_evidence",
            "failures": collect_failures(structural),
            "results": structural,
        },
        "pythia_precision_ablation": {
            "role": "secondary_dtype_precision_ablation",
            "primary_losslessness_test": False,
            "failures": collect_failures(ablation),
            "results": ablation,
        },
    }
    validate_counts(evidence, len(cases), args.smoke_limit)
    write_json(run_root / "summary.json", evidence)
    evidence_root.mkdir(parents=True)
    write_json(evidence_root / "summary.json", evidence)
    write_json(evidence_root / "evidence.json", evidence)
    (evidence_root / "table.md").write_text(render_markdown(evidence), encoding="utf-8")
    (evidence_root / "table.tex").write_text(render_latex(evidence), encoding="utf-8")
    (evidence_root / "paper_text.md").write_text(render_paper_text(evidence), encoding="utf-8")
    (evidence_root / "paper_text.tex").write_text(render_paper_text(evidence, latex=True), encoding="utf-8")
    print(f"COMPLETE: {evidence_root.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
