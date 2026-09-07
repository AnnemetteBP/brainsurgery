#!/usr/bin/env python
"""Lay the implementations of one test side by side across conditions, agents and tiers.

    .venv/bin/python usability_tests/compare_artifacts.py T1 --target gpt-2 [--agent fable51] [--effort high] \
        [--repeat 1] [--out FILE.md]

Writes a Markdown document with, per run: agent, tier, condition, grade, wall
clock, cost, executions, artifact size (non-blank, non-comment lines), the
tools the participant reported (condition F), and the full artifact text
(solution.py, plan.yaml, run.sh, and any other file the participant authored
under out/<test>/ except the checkpoint and the self-report). Default output:
log/compare-<test>-<target>.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ARTIFACT_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".json", ".toml", ".txt"}
SKIP = {"REPORT.md", "summary.yaml", "model.safetensors.index.json"}


def loc(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def tools_used(report: Path) -> str:
    if not report.exists():
        return ""
    lines = report.read_text(errors="ignore").splitlines()
    out, grab = [], False
    for line in lines:
        if line.lower().startswith("- tools used"):
            grab = True
            out.append(line.split(":", 1)[1].strip())
            continue
        if grab:
            if line.startswith("- ") or not line.strip():
                break
            out.append(line.strip())
    return " ".join(x for x in out if x)[:300]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test")
    parser.add_argument("--target", required=True)
    parser.add_argument("--agent", action="append", default=None)
    parser.add_argument("--effort", action="append", default=None)
    parser.add_argument("--condition", action="append", default=None, choices=["P", "F", "B"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    runs = []
    for run_json in sorted(HERE.glob(f"*/{args.target}/*/{args.test}-*-{args.repeat}/run.json")):
        r = json.loads(run_json.read_text())
        if args.agent and r["agent"] not in args.agent:
            continue
        if args.effort and r["effort"] not in args.effort:
            continue
        if args.condition and r["condition"] not in args.condition:
            continue
        d = run_json.parent
        h = json.loads((d / "harness.json").read_text()) if (d / "harness.json").exists() else {}
        g = json.loads((d / "grade.json").read_text()) if (d / "grade.json").exists() else {}
        out = d / "out" / args.test
        files = sorted(p for p in out.rglob("*") if p.is_file() and p.suffix in ARTIFACT_SUFFIXES and p.name not in SKIP)
        runs.append((r, h, g, d, files))
    if not runs:
        print("no runs found"); return 1

    order = {"P": 0, "F": 1, "B": 2}
    runs.sort(key=lambda x: (x[0]["agent"], ["low", "light", "medium", "high"].index(x[0]["effort"]), order[x[0]["condition"]]))
    lines = [f"# {args.test} on {args.target}, repeat {args.repeat}: implementations side by side", "",
             "| agent | tier | cond | grade | wall s | cost USD | executions | lines | tools (F) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r, h, g, d, files in runs:
        total_loc = sum(loc(f.read_text(errors="ignore")) for f in files)
        lines.append(f"| {r['agent']} | {r['effort']} | {r['condition']} | {'PASS' if g.get('passed') else 'FAIL'} | "
                     f"{h.get('wall_clock_s', '')} | {h.get('cost_usd', 0) or 0:.2f} | {h.get('executions', '')} | {total_loc} | "
                     f"{tools_used(d / 'out' / args.test / 'REPORT.md') if r['condition'] == 'F' else ''} |")
    lines.append("")
    for r, h, g, d, files in runs:
        lines.append(f"## {r['agent']} / {r['effort']} / {r['condition']}  ({d.relative_to(HERE)})")
        for f in files:
            text = f.read_text(errors="ignore")
            fence = "python" if f.suffix == ".py" else ("yaml" if f.suffix in (".yaml", ".yml") else ("bash" if f.suffix == ".sh" else ""))
            lines += [f"### {f.relative_to(d)}  ({loc(text)} lines)", "", f"```{fence}", text.rstrip(), "```", ""]
    out_path = args.out or REPO / "log" / f"compare-{args.test}-{args.target}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
