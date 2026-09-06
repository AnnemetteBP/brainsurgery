#!/usr/bin/env python
"""Redo the bug-detection review of every graded Claude cell under one protocol.

    .venv/bin/python usability_tests/review_pass.py [--agents sonnet5 opus5 fable51] [--repeats 1 2] \
        [--parallel 4] [--max-attempts 3]

For each cell, runs `run_claude.py --review-only` (the previous review is kept
as review-attempt-<n>.json). A reply without a YES/NO verdict, or a rate or
session limit, is retried after a wait. Model ids per agent directory come
from --models (agent=model pairs) with the study defaults. Resumable: cells
whose review.json already carries `protocol` equal to --protocol are skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MODELS = {"sonnet5": "claude-sonnet-5", "opus5": "claude-opus-5", "fable51": "claude-fable-5-1"}


def review_cell(cell: Path, model: str, args) -> str:
    agent, target, effort, run = cell.parts[-4:]
    test, cond, rep = run.split("-")
    tag = f"{agent}/{target}/{effort}/{run}"
    rv = cell / "review.json"
    if rv.exists() and json.loads(rv.read_text()).get("protocol") == args.protocol:
        return f"skip   {tag}"
    cmd = [sys.executable, str(HERE / "run_claude.py"), test, cond, "--agent", agent, "--model", model,
           "--target", target, "--effort", effort, "--repeat", rep, "--review-only"]
    for attempt in range(1, args.max_attempts + 1):
        subprocess.run(cmd, cwd=REPO, capture_output=True)
        if not rv.exists():
            time.sleep(args.wait_s)
            continue
        r = json.loads(rv.read_text())
        text = r.get("verdict_text") or ""
        if r.get("verdict") in ("yes", "no"):
            r["protocol"] = args.protocol
            rv.write_text(json.dumps(r, indent=2) + "\n")
            return f"done   {tag} verdict={r['verdict']} ({r.get('artifact_kind')})"
        time.sleep(args.limit_wait_s if "limit" in text.lower() else args.wait_s)
    return f"FAIL   {tag} no verdict after {args.max_attempts} attempts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--models", nargs="*", default=[], help="agent=model overrides")
    parser.add_argument("--repeats", nargs="+", default=["1", "2"])
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--wait-s", type=int, default=20)
    parser.add_argument("--limit-wait-s", type=int, default=900)
    parser.add_argument("--protocol", default="review-v2", help="tag written into review.json when done")
    args = parser.parse_args()
    models = dict(DEFAULT_MODELS)
    for pair in args.models:
        k, v = pair.split("=", 1)
        models[k] = v
    cells = [c for a in args.agents for c in sorted(HERE.glob(f"{a}/*/*/*"))
             if c.is_dir() and (c / "grade.json").exists() and c.name.split("-")[-1] in args.repeats]
    print(f"{len(cells)} cells, parallel={args.parallel}", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(review_cell, c, models[c.parts[-4]], args) for c in cells]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)
    print("review pass done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
