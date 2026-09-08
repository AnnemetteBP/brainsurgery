# Handover: finishing the usability study

State at 2026-09-07, branch `main`. One piece of work remains: 139 Fable 5.1
re-reviews on this machine, blocked on a weekly account limit (section 3).
The Codex cohort is complete. Everything else is done; the Codex cells and
this update are not yet committed.

## 1. What is already finished

| Piece | State |
|---|---|
| Kit (tasks, conditions, references, grader, drivers) | complete, verified on all three targets |
| Solve phase, Claude agents | 810 cells (3 agents x 3 targets x 3 tiers x 5 tests x 3 conditions x 2 repeats), **all pass**, no cap hits, 580.80 USD |
| Reviews, Sonnet 5 and Opus 5 | complete under the final protocol (`review-v2`) against the final artifacts |
| Reviews, Fable 5.1 | **139 outstanding** (130 against superseded artifacts, 9 blocked by the account limit); see section 3 |
| Codex runs | complete: `sol_eacl2027`, model `gpt-5.6-sol`, 270 cells over both repeats, all pass, no cap hits, 104.60 USD imputed |
| Doc-consultation time | derived from the transcripts for all 810 Claude cells and committed as `doctime.json`; results in `README.md` |

Records per cell: `run.json`, `harness.json`, `grade.json`, `review.json`,
`doctime.json`, `env-freeze.txt`, the participant's artifact under
`out/<test>/` and its `REPORT.md`. `analyze.py` reads only the JSON files.

## 2. Uncommitted work in the tree

`git status` shows ~400 modified `review.json` files: the Sonnet 5 and Opus 5
re-reviews plus the `artifact_sha256` backfill. Commit them before anything
else, or they are lost:

```bash
git add usability_tests/sonnet5 usability_tests/opus5 usability_tests/fable51
git commit -m "usability_tests: reviews under the final protocol (sonnet5, opus5)"
```

## 3. Finishing the Fable 5.1 reviews

**State confirmed 2026-09-07.** 139 of the 270 fable51 cells still need their
review redone. Nothing about the solve phase is affected: all 270 fable51
cells solved successfully (`grade.json` `passed=true`, no cap hits), and every
participant-side measure (success, time, retries, tokens, cost) is final. Only
the bug-detection review is outstanding, and only for fable51 -- opus5 and
sonnet5 are 270/270 clean.

The 139 split into two causes:

| n | cause | `review.json` state |
|---|---|---|
| 130 | reviewed against artifact text that commit `a9d721c8` (2026-09-06 18:48) then rewrote | `protocol: review-v2`, real verdict, but `artifact_sha256` no longer matches the file on disk |
| 9 | the Fable account limit was reached mid-review | `protocol: null`, `verdict_text` is literally `"You've reached your Fable limit..."`, 3-4 `review-attempt-*.json` already beside it |

All 130 stale ones are condition **P**: `a9d721c8` made the Python references
self-contained, so 36 distinct artifacts under `review/*/P/` and
`solutions/*/P/` changed. A review is only valid for the artifact text it
read, which is why `review_pass.py` compares `artifact_sha256`.

The 9 limit-blocked cells, for reference:

```
fable51/gpt-2/low/T1-P-1
fable51/pythia-1b/high/T1-P-1   T1-P-2   T2-P-1   T2-P-2
fable51/pythia-1b/high/T3-P-1   T3-P-2   T4-P-1   T4-P-2
```

### Why this matters before the tables are written

`analyze.py` currently reports fable51 at a 5.3% false-alarm rate over n=131,
but 130 of the counted verdicts were formed against superseded artifacts and
9 are excluded outright. **That number is not comparable to the other agents
until this pass is redone.** opus5 (0.0%), sonnet5 (17.8%) and sol_eacl2027
(23.7%) are all computed from current artifacts. Bug detection is 100% for
every agent, so the false-alarm column is the one that carries the result.

### Precondition: has the limit reset?

The waiter from 2026-09-06 (`log/fable-rereview-waiter.txt`) probed six times
between 20:48 and 22:03 and got "still limited" every time; it is no longer
running. Probe by hand:

```bash
claude -p "Reply with the single word OK." --model claude-fable-5-1
```

A plain `OK` means go. A limit message means wait -- it is a weekly cap, so
check once a day rather than looping.

### The run

```bash
cd /home/rootkidd/Projects/brainsurgery
.venv/bin/python usability_tests/review_pass.py --agents fable51 --parallel 4 --max-attempts 4
```

No `--force` is needed and it must not be used: `review_cell()` skips a cell
only when `protocol == review-v2` **and** `artifact_sha256` matches the file
on disk, so exactly the 139 are redone and the 131 good ones are left alone.
Each cell's previous review is preserved as `review-attempt-<n>.json`. On a
reply containing "limit" the driver waits `--limit-wait-s` (900 s) before
retrying, so a partial reset will stall rather than corrupt anything; re-run
the same command to resume. Budget roughly 20 minutes and ~20 USD.

### Verify, then re-derive

```bash
# expect: fable51 drops out entirely (sol_eacl2027's 270 are a known false
# positive -- run_codex.py writes no protocol/verdict/artifact_sha256 field)
.venv/bin/python - <<'EOF'
import json, hashlib
from pathlib import Path
HERE = Path("usability_tests"); bad = {}
for c in HERE.glob("*/*/*/*"):
    rv = c / "review.json"
    if not (c / "grade.json").exists() or not rv.exists(): continue
    r = json.loads(rv.read_text()); art = HERE / (r.get("artifact") or "")
    cur = hashlib.sha256(art.read_bytes()).hexdigest() if art.exists() else None
    if r.get("protocol") != "review-v2" or r.get("verdict") not in ("yes","no") \
       or r.get("artifact_sha256") != cur:
        bad[c.parts[1]] = bad.get(c.parts[1], 0) + 1
print("stale reviews by agent:", bad)
EOF

.venv/bin/python usability_tests/resummarise.py     # execution counts from transcripts
.venv/bin/python usability_tests/doc_time.py        # doc-consultation time (Claude transcripts only)
.venv/bin/python usability_tests/analyze.py         # per agent/target/effort/condition + pooled
```

Then re-archive the transcripts, since a re-review writes new ones and the
existing snapshot predates them:

```bash
find usability_tests -name transcript.jsonl -print0 | sort -z > /tmp/tx.list
tar --zstd -cf /mnt/nvme/brainsurgery/log/usability_tests-transcripts-<date>.tar.zst \
    --null -T /tmp/tx.list
```

Finally write the tables into `usability_tests/README.md` ("Results, repeat 1"
still holds repeat-1-only numbers) and into the PR description at
`/mnt/nvme/brainsurgery/log/PR-usability-study.md`.

### Traps specific to this pass

- `pgrep -f run_full_codex.sh` and similar self-match the shell running the
  check. Match on `'^[0-9]+ bash usability_tests/...'` instead.
- The 9 limit-blocked cells already carry 3-4 `review-attempt-*.json` files.
  That is expected history, not corruption; do not delete them.
- `review.detected` is null and `error_class` empty across all 1080 cells in
  every cohort. That is the intended state -- `analyze.py` derives detection
  from `verdict_text` -- not outstanding work. Filling them for one agent only
  would break parity.

## 4. Running Codex

Prerequisites: `codex` CLI on PATH, logged into an account with an active
subscription, and the study data verified on the machine that runs it.

```bash
.venv/bin/python usability_tests/make_manifest.py --verify   # must be 126/126
```

If it is not 126/126, do not regenerate: the seeded inputs are not
bit-identical across hardware. Transfer the bundle instead
(`usability_tests/pack_data.sh` made `/mnt/nvme/brainsurgery/log/usability_tests-data.tar`,
50.6 GB, sha256 `7001fedf486d026eea60213108278e4ddba41de40cb5b4a6d0db28f4dbcd28fe`),
extract into `models/`, rerun `setup.py`, verify again. Base models must be at
the pinned revisions in `targets.py`.

One cell first, then check the parsing against the transcript:

```bash
.venv/bin/python usability_tests/run_codex.py T1 P --agent <name> --model <model-id> \
    --target gpt-2 --effort light --reasoning-effort low --repeat 1 --venv \
    --price-in <usd/M> --price-out <usd/M>
.venv/bin/python usability_tests/audit_codex.py            # teammate's audit helper
```

Then the full repeat:

```bash
AGENT=<name> MODEL=<model-id> PRICE_IN=<usd> PRICE_OUT=<usd> \
  PRICE_CACHE_READ=<usd> PRICE_CACHE_WRITE=<usd> \
  nohup usability_tests/run_full_codex.sh 1 2 > log/usability-codex-full-r1.txt 2>&1 &
```

Repeat 1 first and completely, then repeat 2 (`... 2 2`). Odd repeats show the
defective artifact to the reviewer, even repeats the correct one; both are
needed for detection and false-alarm rates. Transcripts are not committed, so
anything derived from them has to be extracted on that machine
(`.venv/bin/python usability_tests/doc_time.py`) before cleanup, but note that
`doc_time.py` reads Claude Code transcripts only and skips Codex ones as
unsupported: extend it against one real Codex transcript if the doc-time
column is wanted for both vendors.

Parity requirements, all in `usability_tests/AGENTS.md` under "Running the same
study with another driver": same commit, verified data, same cells, sandboxes
from `make_sandbox.py --venv`, same phases and record fields, 30-minute cap,
error classes filled in, `detected` confirmed against
`review/<target>/answers.json`, cost from the rate card.

Known differences to report rather than hide: Codex has no turn cap, only the
time cap, so `cap_hit` values are not directly comparable; the tier names are
`light`/`medium`/`high` and map to reasoning effort `low`/`medium`/`high`.

## 5. Traps already hit, so they are not hit again

- **Do not regenerate inputs on another machine.** QR and matrix products
  differ by CPU/BLAS; only 34 of 126 files matched. Transfer the bundle.
- **Inputs are read-only** (`setup.py` enforces it) after a participant wrote
  through a copied symlink into the shared GPT-2 base and invalidated a cell.
- **Condition B installs BrainSurgery non-editable**; an editable install let
  participants read the repository source.
- **Rate and session limits** come back as a result with `is_error` true; the
  runners discard the attempt and retry. Fable's limit is per model.
- **Scratch space**: OLMo outputs are 4.7 GB per cell, and `/tmp` is a 39 GB
  tmpfs. Verification scratch belongs on `/mnt/nvme/brainsurgery/log/`.
- **Sandboxes are deleted after grading** (`--keep-artifacts` to keep them);
  the pilot's 45 cells used 140 GB, the study data is 107 MB.
- **Transcripts are gitignored**, so a measure derived from one is lost when
  the machine is cleaned. `doc_time.py` writes `doctime.json` per cell;
  run it before cleanup, and on every machine that drives an agent.
- **A review is only valid for the artifact it read.** If any reference or
  defective variant changes, `review_pass.py` will redo exactly the affected
  reviews; never edit those files without rerunning it.

## 6. Open decisions for the paper

- Detection is 100% in every condition, so that half does not discriminate;
  harder defects would be needed for separation.
- In 4 of 10 plan false alarms the verdict text contradicts its own first
  word. Scored on the first word (pre-registered rule), but worth a footnote.
- Sonnet 5 raises every false alarm; Opus 5 raises none. Report per model.
- T2 is longer as a plan than as a script because `concat` takes single tensor
  references; a bulk form would change that number.
