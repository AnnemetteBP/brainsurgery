#!/usr/bin/env python
"""Derive doc-consultation time from the run transcripts.

    .venv/bin/python usability_tests/doc_time.py [--root DIR] [--force]

Reads `transcript.jsonl` in every run directory and writes `doctime.json` next
to it, so the measure survives the transcript (transcripts are gitignored and
are deleted when a machine is cleaned; the derived record is committed).
`analyze.py` reads `doctime.json` and reports it per group.

Measure. Every message in the transcript carries a millisecond timestamp. For
each tool call that touches the condition's documentation surface, the span
runs from the `tool_use` timestamp to the first assistant message after the
tool result, i.e. fetching the text plus the model turn that consumes it.
Overlapping spans are merged, so a chain of reads is counted once.

    doc_time_s        upper bound: the span as described
    doc_time_lower_s  lower bound: the same, except that a span whose closing
                      message performs a non-documentation action contributes
                      only the tool latency, since that message's generation
                      time is not attributable to reading

The two bounds are within about 1% of each other in condition B (page 3 of
the reasoning: only 41 of 1968 doc reads are closed by an action message, and
no assistant message ever mixed a doc read with other tool calls, so the
attribution is not confounded by batching). `spans_closed_by_action` and
`mixed_messages` are recorded per run so that stays checkable.

Documentation surface. In condition B it is the doc pack, addressed by path
(`docpack/...`) whether read with Bash or with the Read tool. Conditions P and
F have no doc pack; what corresponds is inspecting an installed package, so
the pattern covers `--help`, `pydoc`, `help()`, `__doc__`,
`inspect.getsource`, `pip show` and any path under `site-packages`. That is a
heuristic and it is near-zero in practice: P and F agents work from what they
already know about torch, peft and the merge toolkits, so the two conditions
carry no comparable doc-reading cost to contrast with B's.

This is agent latency, not human reading time. It compares conditions, agents
and effort tiers with each other; it does not estimate what a practitioner
would spend.

Transcript format. `doctime-v1` reads the Claude Code stream-json transcript
that `run_claude.py` writes: assistant and user messages carrying `timestamp`
and `tool_use`/`tool_result` blocks. `run_codex.py` writes the raw
`codex exec --json` event stream instead, which this parser does not
understand, so a Codex run is reported as unsupported and no `doctime.json` is
written for it rather than a misleading zero. Extending the measure to Codex
means reading one real Codex transcript, checking that its events carry
timestamps and command text, and adding a second reader here.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = "doctime-v1"

DOC_PATTERN = {
    "B": re.compile(r"docpack"),
    "other": re.compile(r"--help|\bpydoc\b|help\(|__doc__|inspect\.getsource|pip show|site-packages"),
}


def stamp(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def events(path: Path) -> list[tuple]:
    """("A", t, [(tool_use_id, input_json), ...]) and ("R", t, tool_use_id, result_size).

    Empty for a transcript that is not Claude Code stream-json.
    """
    out = []
    for line in path.read_text().splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        when = message.get("timestamp")
        if not when:
            continue
        if message.get("type") == "assistant":
            content = message.get("message", {}).get("content", []) or []
            uses = [(c["id"], json.dumps(c["input"])) for c in content if c.get("type") == "tool_use"]
            out.append(("A", stamp(when), uses))
        elif message.get("type") == "user":
            content = message.get("message", {}).get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "tool_result":
                        out.append(("R", stamp(when), c.get("tool_use_id"), len(json.dumps(c.get("content")))))
    out.sort(key=lambda e: e[1])
    return out


def measure(transcript: Path, condition: str) -> dict | None:
    """None if the transcript is in a format this measure cannot read."""
    pattern = DOC_PATTERN.get(condition, DOC_PATTERN["other"])
    timeline = events(transcript)
    if not any(e[0] == "A" for e in timeline):
        return None
    assistant = [(e[1], e[2]) for e in timeline if e[0] == "A"]
    result_at = {e[2]: e[1] for e in timeline if e[0] == "R"}
    result_size = {e[2]: e[3] for e in timeline if e[0] == "R"}

    upper_spans: list[tuple[float, float]] = []
    lower_spans: list[tuple[float, float]] = []
    calls = doc_bytes = mixed = closed_by_action = 0
    for event in timeline:
        if event[0] != "A":
            continue
        uses = event[2]
        docs = [u for u in uses if pattern.search(u[1])]
        if not docs:
            continue
        calls += len(docs)
        if len(docs) != len(uses):
            mixed += 1
        doc_bytes += sum(result_size.get(uid, 0) for uid, _ in docs)
        ends = [result_at[uid] for uid, _ in docs if uid in result_at]
        if not ends:
            continue
        start, end = event[1], max(ends)
        following = next(((t, uses) for t, uses in assistant if t > end), None)
        if following is None:
            upper_spans.append((start, end))
            lower_spans.append((start, end))
            continue
        # the closing message digests the text if it only reads more of it, or
        # only thinks or speaks; otherwise its generation time is a mix
        next_t, next_uses = following
        digesting = not next_uses or all(pattern.search(i) for _, i in next_uses)
        if not digesting:
            closed_by_action += 1
        upper_spans.append((start, next_t))
        lower_spans.append((start, next_t if digesting else end))

    return {
        "protocol": PROTOCOL,
        "source": "transcript.jsonl",
        "condition": condition,
        "doc_calls": calls,
        "doc_bytes": doc_bytes,
        "doc_time_s": round(merge(upper_spans), 1),
        "doc_time_lower_s": round(merge(lower_spans), 1),
        "spans_closed_by_action": closed_by_action,
        "mixed_messages": mixed,
    }


def merge(spans: list[tuple[float, float]]) -> float:
    total = 0.0
    current: tuple[float, float] | None = None
    for start, end in sorted(spans):
        if current and start <= current[1]:
            current = (current[0], max(current[1], end))
            continue
        if current:
            total += current[1] - current[0]
        current = (start, end)
    if current:
        total += current[1] - current[0]
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--force", action="store_true", help="recompute runs that already have doctime.json")
    args = parser.parse_args()

    written = skipped = missing = unsupported = 0
    for run_json in sorted(args.root.glob("*/*/*/*/run.json")):
        sandbox = run_json.parent
        record = sandbox / "doctime.json"
        if record.exists() and not args.force:
            skipped += 1
            continue
        transcript = sandbox / "transcript.jsonl"
        if not transcript.exists():
            missing += 1
            continue
        condition = (json.loads(run_json.read_text()).get("condition")) or sandbox.name.split("-")[1]
        result = measure(transcript, condition)
        if result is None:
            print(f"[doc_time] unreadable transcript format, skipped: {sandbox}")
            unsupported += 1
            continue
        record.write_text(json.dumps(result, indent=1) + "\n")
        written += 1
    print(f"doctime.json: {written} written, {skipped} already present, "
          f"{missing} without a transcript, {unsupported} in an unsupported format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
