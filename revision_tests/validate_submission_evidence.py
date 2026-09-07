#!/usr/bin/env python3
"""Audit EACL 2027 evidence without running any experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
LINUX_COMMIT = "2dbcd505115100f892e906413076ae93b3fcaa16"


@dataclass(frozen=True)
class Check:
    item: str
    status: str
    detail: str


def load(path: str) -> dict[str, Any]:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reachable(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def correctness() -> Check:
    fixture = load("revision_tests/correctness/results/2026_09_06_macos_30adb808/summary.json")
    real = load("revision_tests/correctness/results/2026_09_06_real_macos_c5e464b9/summary.json")
    passed = (
        fixture["passed"]
        and real["passed"]
        and fixture["cases_passed"] == fixture["cases_total"] == 10
        and real["cases_passed"] == real["cases_total"] == 3
        and fixture["oracle_tensors_exact"] == fixture["oracle_tensors_checked"] == 161
        and real["tensors_exact"] == real["tensors_checked"] == 518
        and fixture["untouched_tensors_exact"] == fixture["untouched_tensors_checked"] == 149
        and real["untouched_tensors_exact"] == real["untouched_tensors_checked"] == 515
        and reachable(fixture["repository_commit"])
        and reachable(real["repository_commit"])
        and (REPO / "revision_tests/correctness/results/paper_table.tex").is_file()
    )
    return Check(
        "Correctness and preservation",
        "PASS" if passed else "BLOCKED",
        "13/13 cases; 679/679 oracle checks; 664/664 untouched checks" if passed else "correctness counts, table, or execution revision failed validation",
    )


def linux_results() -> list[Check]:
    robust = load("revision_tests/robustness/results/linux_2dbcd50/summary.json")
    scaling = load("revision_tests/scaling/results/linux_2dbcd50/summary.json")
    tools = load("revision_tests/competing_tools/results/linux_2dbcd50/summary.json")
    commit_ok = reachable(LINUX_COMMIT)
    robust_numbers = (
        robust["evaluation_passed"]
        and robust["cases_passed"] == robust["cases_total"] == 19
        and robust["sources_unchanged"] == 19
        and robust["observed_safe_cases"] == 16
        and robust["partial_or_mixed_output_findings"] == 3
        and robust["repository_commit"] == LINUX_COMMIT
    )
    scaling_numbers = (
        scaling["reported_eligible"]
        and scaling["measured_attempts"] == scaling["correct_measured_attempts"] == 150
        and len(scaling["models"]) == 10
        and len(scaling["pairs"]) == 30
        and scaling["git_commit"] == LINUX_COMMIT
    )
    tool_numbers = (
        tools["reported_eligible"]
        and tools["measured_attempts"] == tools["correct_measured_attempts"] == 30
        and len(tools["comparisons"]) == 3
        and len(tools["pairs"]) == 6
        and tools["git_commit"] == LINUX_COMMIT
    )

    def result(name: str, numbers: bool, detail: str) -> Check:
        if not numbers:
            return Check(name, "BLOCKED", "measurement summary failed validation")
        if not commit_ok:
            return Check(name, "BLOCKED", f"numbers validated; execution commit {LINUX_COMMIT} is not reachable")
        return Check(name, "PASS", detail)

    return [
        result("Robustness and failure semantics", robust_numbers, "19/19 cases; 19/19 sources unchanged; 16/19 safe destinations; three negative mid-save findings"),
        result("Scaling and systems", scaling_numbers, "10 checkpoints; 30 model-method pairs; 150/150 outputs correct"),
        result("Competing tools", tool_numbers, "three matched operations; six tool-case pairs; 30/30 outputs correct"),
    ]


def environment_records() -> Check:
    paths = [
        "revision_tests/scaling/results/linux_2dbcd50/environment.json",
        "revision_tests/competing_tools/results/linux_2dbcd50/environment.json",
        "revision_tests/robustness/results/linux_2dbcd50/environment.json",
        "revision_tests/plans/paper_environment.tex",
    ]
    absent = [path for path in paths if not (REPO / path).is_file()]
    return Check(
        "Linux run provenance files",
        "PASS" if not absent else "BLOCKED",
        "all sanitized records and paper paragraph present" if not absent else "missing: " + ", ".join(absent),
    )


def behavioral() -> Check:
    protocol_path = REPO / "revision_tests/behavioral/paper_protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol_id = protocol["protocol_id"]
    expected_ids = protocol["expected_model_ids"]
    prompts_per_model = protocol["inference"]["prompts_per_model"]
    required_comparisons = set(protocol["reporting"]["required_comparisons"])
    required_metrics = set(protocol["required_old_paper_metrics"]) | set(protocol.get("additional_metrics", []))
    aggregate_fields = {
        "prompt_count", "reference_mean_perplexity", "transformed_mean_perplexity",
        "mean_perplexity_ratio", "maximum_perplexity_ratio",
        "mean_last_token_logit_cosine", "minimum_last_token_logit_cosine",
        "mean_last_token_absolute_difference", "maximum_last_token_absolute_difference",
        "mean_full_sequence_logit_cosine", "minimum_full_sequence_logit_cosine",
        "full_sequence_positions_compared", "mean_full_sequence_absolute_difference",
        "maximum_full_sequence_absolute_difference", "top1_matches",
        "output_exact_matches", "mean_output_char_similarity",
        "mean_output_token_sequence_similarity", "mean_output_token_bag_cosine",
        "mcq_count", "mcq_prediction_matches",
    }
    group_by = set(protocol["reporting"]["group_by"])

    def valid_comparison(comparison: dict[str, Any]) -> bool:
        rows = comparison.get("prompts", [])
        aggregate = comparison.get("aggregate", {})
        breakdowns = comparison.get("by_manifest_dimension", {})
        return (
            len(rows) == prompts_per_model
            and all(required_metrics <= row.keys() for row in rows)
            and aggregate_fields <= aggregate.keys()
            and aggregate.get("prompt_count") == prompts_per_model
            and set(breakdowns) == group_by
            and all(
                sum(group.get("prompt_count", 0) for group in breakdowns[field].values())
                == prompts_per_model
                and all(aggregate_fields <= group.keys() for group in breakdowns[field].values())
                for field in group_by
            )
        )

    def valid_result(result: dict[str, Any]) -> bool:
        comparisons = result.get("comparisons", {})
        oracles = result.get("tensor_oracle", {})
        return (
            set(comparisons) == required_comparisons
            and all(valid_comparison(comparisons[name]) for name in required_comparisons)
            and set(oracles) == {
                "brainsurgery_scaled", "python_scaled", "brainsurgery_restored"
            }
            and all(
                oracle.get("passed") is True
                and oracle.get("tensors_checked", 0) > 0
                and oracle.get("tensors_passed") == oracle.get("tensors_checked")
                for oracle in oracles.values()
            )
            and oracles["brainsurgery_restored"].get("reference")
            == protocol["reporting"]["restored_tensor_reference"]
        )

    root = REPO / "revision_tests/behavioral/results"
    candidates = []
    for evidence_path in root.glob("*/evidence.json"):
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("protocol_id") == protocol_id:
            candidates.append((evidence_path.parent, evidence))
    for directory, evidence in candidates:
        prompt_total = sum(
            row["comparisons"]["imperative_equivalence"]["aggregate"]["prompt_count"]
            for row in evidence.get("results", [])
        )
        required = ["evidence.json", "table.md", "table.tex", "paper_text.md", "paper_text.tex"]
        complete = (
            evidence.get("protocol_sha256") == sha256(protocol_path)
            and [row.get("id") for row in evidence.get("results", [])] == expected_ids
            and prompt_total == len(expected_ids) * prompts_per_model
            and all(valid_result(row) for row in evidence.get("results", []))
            and all((directory / name).is_file() for name in required)
            and reachable(evidence.get("git_commit", ""))
        )
        if complete:
            return Check("Expanded behavioral analysis", "PASS", f"{directory.relative_to(REPO)}: 10 models and 700 prompt-model comparisons")
        missing = [name for name in required if not (directory / name).is_file()]
        return Check(
            "Expanded behavioral analysis",
            "BLOCKED",
            f"{directory.relative_to(REPO)} matches {protocol_id}, but is incomplete"
            + (": missing " + ", ".join(missing) if missing else ""),
        )
    return Check("Expanded behavioral analysis", "PENDING", f"no evidence bundle matches canonical protocol {protocol_id}")


def usability() -> Check:
    official = REPO / "usability_tests/astra_eacl2027"
    if not official.is_dir():
        return Check("Coding-agent usability and auditability", "PENDING", "official Codex namespace is absent and complete audited cohorts are not present")
    records = list(official.rglob("run.json"))
    complete = all(
        all((path.parent / name).is_file() for name in ("harness.json", "grade.json", "review.json"))
        for path in records
    )
    return Check(
        "Coding-agent usability and auditability",
        "PENDING",
        f"{len(records)} official Codex cells have record quartets" if complete else "one or more official Codex cells lack required records",
    )


def manuscript() -> Check:
    tex = list((REPO / "private/EACL2027_submission").glob("*.tex"))
    return Check(
        "Final manuscript and demo",
        "PENDING",
        "manuscript sources found; claim audit and demo still require completion" if tex else "no EACL manuscript .tex is present here; only plans and fragments exist",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="return nonzero unless every required gate passes")
    args = parser.parse_args()
    checks = [correctness(), *linux_results(), environment_records(), behavioral(), usability(), manuscript()]
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        print("| Evidence gate | Status | Detail |")
        print("|---|---|---|")
        for check in checks:
            print(f"| {check.item} | **{check.status}** | {check.detail} |")
    ready = all(check.status == "PASS" for check in checks)
    return 1 if args.strict and not ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
