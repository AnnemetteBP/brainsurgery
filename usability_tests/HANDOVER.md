# Handover: finishing the usability study

State at 2026-09-06 22:00, branch `usability-study`, last commit `a96fda78`.
Two pieces of work remain: 138 blocked Fable 5.1 re-reviews on this machine,
and the Codex runs. Everything else is done and committed.

## 1. What is already finished

| Piece | State |
|---|---|
| Kit (tasks, conditions, references, grader, drivers) | complete, verified on all three targets |
| Solve phase, Claude agents | 810 cells (3 agents x 3 targets x 3 tiers x 5 tests x 3 conditions x 2 repeats), **all pass**, no cap hits, 580.80 USD |
| Reviews, Sonnet 5 and Opus 5 | complete under the final protocol (`review-v2`) against the final artifacts |
| Reviews, Fable 5.1 | **138 stale**, blocked by a model-specific account limit |
| Codex runs | 3 cells exist (`usability_tests/astra/gpt-2/light/T1-*-1`), the rest not run |

Records per cell: `run.json`, `harness.json`, `grade.json`, `review.json`,
`env-freeze.txt`, the participant's artifact under `out/<test>/` and its
`REPORT.md`. `analyze.py` reads only the JSON files.

## 2. Uncommitted work in the tree

`git status` shows ~400 modified `review.json` files: the Sonnet 5 and Opus 5
re-reviews plus the `artifact_sha256` backfill. Commit them before anything
else, or they are lost:

```bash
git add usability_tests/sonnet5 usability_tests/opus5 usability_tests/fable51
git commit -m "usability_tests: reviews under the final protocol (sonnet5, opus5)"
```

## 3. Finishing the Fable 5.1 reviews

Why they are stale: the reference artifacts changed after those reviews ran
(the Python baselines became self-contained, T3/T4/T5 plans got literal
required checks). A review is only valid for the artifact text it read, so
`review_pass.py` compares `artifact_sha256` in `review.json` against the file
on disk and redoes a cell when they differ.

A waiter is running (`log/fable-rereview-waiter.txt`); it probes the limit
every 15 minutes and starts the re-review automatically. If it is gone, do it
by hand once `claude -p "Reply with the single word OK." --model claude-fable-5-1`
answers without a limit message:

```bash
.venv/bin/python usability_tests/review_pass.py --agents fable51 --parallel 4 --max-attempts 4
```

Roughly 138 reviews, about 20 minutes, about 20 USD. Then check nothing is
stale and commit:

```bash
.venv/bin/python - <<'PY'
import json, hashlib
from pathlib import Path
HERE = Path("usability_tests"); stale = 0
for c in HERE.glob("*/*/*/*"):
    rv = c / "review.json"
    if not (c / "grade.json").exists() or not rv.exists(): continue
    r = json.loads(rv.read_text()); art = HERE / (r.get("artifact") or "")
    cur = hashlib.sha256(art.read_bytes()).hexdigest() if art.exists() else None
    stale += r.get("protocol") != "review-v2" or r.get("verdict") not in ("yes","no") or r.get("artifact_sha256") != cur
print("stale reviews:", stale)
PY
```

Then the final tables:

```bash
.venv/bin/python usability_tests/resummarise.py     # execution counts from transcripts
.venv/bin/python usability_tests/analyze.py         # per agent/target/effort/condition + pooled
```

and write them into `usability_tests/README.md` ("Results, repeat 1" section,
which currently holds repeat-1-only numbers) and into the PR description at
`/mnt/nvme/brainsurgery/log/PR-usability-study.md`.

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
needed for detection and false-alarm rates.

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
