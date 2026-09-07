#!/usr/bin/env python3
"""Audit EACL 2027 evidence without running any experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
LINUX_COMMIT = "2dbcd505115100f892e906413076ae93b3fcaa16"


@dataclass(frozen=True)
class Check:
    item: str
    status: str
    detail: str


def load(path: str) -> dict[str, Any]:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


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
    root = REPO / "revision_tests/behavioral/results"
    candidates = []
    for evidence_path in root.glob("*/evidence.json"):
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("protocol_id") == "eacl2027_behavioral_paper_v3":
            candidates.append((evidence_path.parent, evidence))
    for directory, evidence in candidates:
        prompt_total = sum(
            row["comparisons"]["imperative_equivalence"]["aggregate"]["prompt_count"]
            for row in evidence.get("results", [])
        )
        required = ["evidence.json", "table.md", "table.tex", "paper_text.md", "paper_text.tex"]
        complete = (
            evidence.get("reported_eligible") is True
            and len(evidence.get("results", [])) == 10
            and prompt_total == 700
            and all((directory / name).is_file() for name in required)
            and reachable(evidence.get("git_commit", ""))
        )
        if complete:
            return Check("Expanded behavioral analysis", "PASS", f"{directory.relative_to(REPO)}: 10 models and 700 prompt-model comparisons")
    return Check("Expanded behavioral analysis", "PENDING", "no complete reportable v3 evidence bundle is present; v2 is excluded")


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
