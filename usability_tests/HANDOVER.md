# Handover: the usability study

State at 2026-09-10, branch `main`. **The study is complete.** All 1080 cells
are solved, graded and reviewed under the final protocol, for all four agents.
Nothing is outstanding. This file is now a record of how it was run and of the
traps worth not hitting again; sections 4 and 5 are the parts still worth
reading before touching the kit.

## 1. Final state

| Piece | State |
|---|---|
| Kit (tasks, conditions, references, grader, drivers) | complete, verified on all three targets |
| Solve phase | 1080 cells (4 agents x 3 targets x 3 tiers x 5 tests x 3 conditions x 2 repeats), **all pass**, no cap hits |
| Reviews | 1080 cells under `review-v2`, every verdict formed against the artifact text on disk |
| Doc-consultation time | `doctime.json` for the 810 Claude cells; Codex transcripts are an unsupported format |
| Cost | 707.35 USD total: 575.28 solve, 132.07 review. The Codex share (121.54) is imputed from the rate card |

Agents: `fable51`, `opus5`, `sonnet5` (Claude Code, tiers `low`/`medium`/`high`)
and `sol_eacl2027` (model `gpt-5.6-sol`, Codex CLI, tiers `light`/`medium`/`high`),
270 cells each. `astra/` holds two excluded macOS pilot cells and is skipped by
`analyze.py`; pass `--include-excluded` to see them.

Records per cell: `run.json`, `harness.json`, `grade.json`, `review.json`,
`doctime.json`, `env-freeze.txt`, the participant's artifact under
`out/<test>/` and its `REPORT.md`. `analyze.py` reads only the JSON files.

## 2. The headline result

Bug detection is 100% for every agent in every condition, so that half of the
review measure does not discriminate. False alarms do, but they separate
models rather than conditions: Opus 5 and Fable 5.1 raise none, Sonnet 5
17.8%, gpt-5.6-sol 23.7%, each roughly flat across P, F and B. All 1080 cells
passed the grader, so correctness does not separate the conditions either. The
measures that do move are effort (a plan costs and takes about twice a Python
script) and Claude first-run success, where condition B leads at 268/270
against 260 for Python. Full tables and caveats are in `README.md`.

## 3. Reproducing the numbers

```bash
.venv/bin/python usability_tests/resummarise.py   # execution counts from Claude transcripts
.venv/bin/python usability_tests/doc_time.py      # doc-consultation time, Claude transcripts only
.venv/bin/python usability_tests/analyze.py       # per agent/target/effort/condition + pooled
```

Both `resummarise.py` and `doc_time.py` skip transcripts they cannot parse and
say how many; a non-zero "unsupported format" count is the Codex cohort and is
expected. Neither writes to a cell whose transcript is missing.

To confirm every review is still valid for the artifact it read:

```bash
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
```

Expected output: `{'astra': 2, 'sol_eacl2027': 270}`. Both are known false
positives -- `astra` is the excluded pilot, and `run_codex.py` writes no
`protocol`/`artifact_sha256` field, so its cells can never satisfy the check.
Any `fable51`, `opus5` or `sonnet5` entry means a reference or defective
variant changed and those reviews must be redone:

```bash
.venv/bin/python usability_tests/review_pass.py --agents <agent> --parallel 4 --max-attempts 4
```

Never pass `--force`. `review_cell()` skips a cell only when its protocol is
`review-v2` and its `artifact_sha256` matches the file on disk, so exactly the
affected cells are redone; each previous review is kept as
`review-attempt-<n>.json`. On a reply containing "limit" the driver waits
`--limit-wait-s` (900 s) and retries, so a partial reset stalls rather than
corrupts. Probe a Claude limit by hand with:

```bash
claude -p "Reply with the single word OK." --model claude-fable-5-1
```

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
- **A derivation script must refuse transcripts it cannot read.**
  `resummarise.py` applied the Claude Code parser to every transcript,
  found nothing in the Codex event stream, and rewrote all 270 valid
  `sol_eacl2027` records to zero executions and `first_execution_success:
  false` -- which showed up only as pooled first-run success dropping from 92%
  to 74%. It now skips a transcript with no `assistant` record and reports the
  count, as `doc_time.py` already did. Any future reader must do the same.
- **`analyze.py` excludes the pilot namespaces in `EXCLUDED_AGENTS`.** Before
  that, `astra`'s two macOS pilot cells were silently pooled into the "all"
  rows, which read 361 runs instead of 360.
- **A review is only valid for the artifact it read.** If any reference or
  defective variant changes, `review_pass.py` will redo exactly the affected
  reviews; never edit those files without rerunning it.

## 6. Open decisions for the paper

- Detection is 100% in every condition for every agent, so that half does not
  discriminate; harder defects would be needed for separation.
- False alarms separate models, not conditions: Opus 5 and Fable 5.1 raise
  none, Sonnet 5 24/135, gpt-5.6-sol 32/135, each roughly flat across P, F
  and B. Report per model, and do not read the review phase as evidence about
  the artifact format.
- In 7 of the 56 false alarms the verdict text concedes after its opening `NO`
  that the behaviour is correct; all 7 are Sonnet 5. Scored on the first word
  (pre-registered rule), but worth a footnote.
- T2 is longer as a plan than as a script because `concat` takes single tensor
  references; a bulk form would change that number.
