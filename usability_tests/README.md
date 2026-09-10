# BrainSurgery usability tests

A usability-study kit for BrainSurgery. Five checkpoint-editing tests, each
instantiated on three surgery-target models and solved under three
conditions by coding agents acting as proxy practitioners. It produces the
evidence the reviewers of the paper ("BrainSurgery: Reproducible and Reliable
Declarative Weight Manipulations for Model Editing and Upcycling") asked for:
success rate, retries, errors, tokens and cost, time to solution, and
bug-detection ability, per test, target, condition and agent. `AGENTS.md`
here is the operating procedure for the agent running the study.

The tests are deliberately not the ones shown in the paper. They are
realistic checkpoint-surgery jobs that exercise the same capability classes
the paper claims (bulk targeting, structural edits, precision casting,
arithmetic, low-rank adapters, validation, sharded export).

## The five tests

| Test | Objective | Capability class | Output |
|---|---|---|---|
| T1 layer-prune | Remove transformer blocks and renumber the rest contiguously | structural edit, bulk rename | single file |
| T2 head-prune | Remove one attention head (a middle one) from every layer, across all head-bearing projections | slicing, concat, layout knowledge | single file |
| T3 mixed-precision-export | Projection matrices to bfloat16, everything else float32, drop buffers, sharded with index | precision cast, bulk targeting, sharding | sharded |
| T4 task-vector-merge | `base + 0.4 (ft1 - base) + 0.4 (ft2 - base)` on the MLP tensors after verifying all other tensors match | multi-checkpoint arithmetic, validation | single file |
| T5 lora-merge | Fold a PEFT-style LoRA adapter (r=16, alpha=32) into the base attention weights | low-rank adapter, matmul, transpose, sharding | sharded |

Each test has a "Why it is meaningful" section in its specification and a
per-target instantiation (`tasks/<test>/TASK-<target>.md`) with concrete
names, shapes, dtypes, counts and layout facts. The specifications are
tool-neutral: they state what must be true of the output and which checks
the solution must perform, and nothing about how to do it.

## Surgery targets

| Target | Model | Layers | Dtype | What makes it different |
|---|---|---|---|---|
| `gpt-2` | GPT-2 (124M), `openai-community/gpt2` | 12 | fp32 | fused `[q k v]` projection, Conv1D `[in, out]` layout, causal-mask buffers, single file |
| `olmo-1b` | OLMo-1B-0724-hf, `allenai/OLMo-1B-0724-hf` | 16 | fp32 | separate q/k/v/o, no biases, no norm parameters, sharded input (two shards + index), 4.7 GB |
| `pythia-1b` | Pythia-1B, `EleutherAI/pythia-1b` | 16 | fp16 | fused QKV interleaved per head (GPT-NeoX), three buffer kinds incl. a uint8 mask, half-precision input that must be upcast for arithmetic |

All facts live in `targets.py`; `generate.py` renders the task texts,
reference solutions, plans and review artifacts from them.

## Conditions

| Code | Tool | Environment | Doc pack |
|---|---|---|---|
| P | Python script with torch and safetensors | `conditions/requirements-P.txt` (pinned) | none |
| F | Free choice from the allowed list: merge toolkits, adapter libraries, HF utilities, key-rewriting tools, or scripts on top of them | `conditions/requirements-F.txt` (pinned lock resolved from the list) | the tools' own docs |
| B | A BrainSurgery YAML plan run with the `brainsurgery` CLI, no Python | the repository installed editable at the recorded commit, `conditions/constraints-B.txt` | `docpack/` |

`conditions/F-allowed.md` is the paper's related-systems list restricted to
what is pip-installable and useful here; `requirements-F.txt` is its resolved
lock. The team may replace both before the pilot. Every task has at least one
plausible route through existing tools (merge toolkits slice layers and do
task arithmetic, transformers prunes heads and saves with a dtype, peft merges
adapters, torch-state-bridge rewrites keys), so F is a real alternative and
not a disguised copy of P.

All three conditions pin torch 2.14.0, safetensors 0.5.3 and numpy 2.5.2 so
runs are comparable across conditions and over the weeks of the study.

## Coding agents and effort tiers

Two drivers produce identical record files, so `analyze.py` treats all
vendors alike:

- `run_claude.py`: Claude models through Claude Code (`claude -p`). Used on
  this machine for Fable 5.1, Opus 5 and Sonnet 5.
- `run_codex.py`: OpenAI models through the Codex CLI (`codex exec --json`).
  The model ids are chosen by whoever runs it; nothing in the kit names one.
  Codex does not report cost, so pass the input, output, cache-read and
  cache-write rates (USD per million tokens), or leave `cost_usd` null for
  the analysis. Used on this machine for gpt-5.6-sol, whose cells are under
  the frozen namespace `sol_eacl2027`. On first use with a new model, run one
  cell and compare `harness.json` with `transcript.jsonl` (see the docstring).

For full Codex runs on Linux, `run_matrix_codex.py` runs one resumable effort
tier and repeat, while `run_full_codex.sh` covers all three effort tiers for
one repeat. The full launcher verifies the data manifest before starting and
defaults to sequential cells. After each batch, `audit_codex.py --agent NAME`
checks every transcript-derived harness field and lists the failed-execution
classes and review decisions that still require experimenter confirmation.

Any other agent can be driven from a copy of this repository with
`make_sandbox.py` plus a driver that writes the same record files (see
`AGENTS.md` and `record-template.md`).

Every agent runs every cell at three effort tiers, using the vendor's own
names: `low`, `medium`, `high` for Claude Code (`claude --effort`), and
`light`, `medium`, `high` for OpenAI models (their reasoning-effort setting).
The tier is a directory level in the results and a field in `run.json` and
`harness.json`, so the analysis reports every measure per tier.

## Where results go

    usability_tests/<agent>/<target>/<effort>/<test>-<condition>-<repeat>/
    usability_tests/sonnet5/gpt-2/medium/T2-B-1/          for example

Agent directory names are short and stable (`fable51`, `opus5`, `sonnet5`,
...). Each run directory is the participant's sandbox and, after the run,
holds that run's study data: `PROMPT.md`, the authored artifact under
`out/<test>/`, the participant's `REPORT.md`, `run.json`, `harness.json`,
`grade.json`, `review.json`, `env-freeze.txt`. `.gitignore` keeps those and
ignores checkpoints, environments, transcripts and copied inputs.

## What is measured

| Measure | Source |
|---|---|
| Success rate | `grade.json` PASS over runs, per cell and pooled |
| Retries, errors | `harness.json`: executions of the script/plan, failed executions with an error class each, first-execution success, executions until first success |
| Tokens and cost | `harness.json`: input/output/cache tokens and cost from the provider (Claude Code reports `total_cost_usd`) |
| Time to solution | `harness.json` wall clock of the solve phase, reported over passing runs |
| Doc-consultation time | `doctime.json`, derived from `transcript.jsonl` by `doc_time.py`: the time spent fetching documentation and digesting it, per run. In condition B that is the doc pack; P and F have no doc pack and the equivalent (`--help`, `pydoc`, reading a package under `site-packages`) is near-zero. Claude Code transcripts only so far |
| Bug-detection ability | `review.json`: after solving, the same model reviews one artifact for the same task (defective on odd repeats, correct on even) and must say whether it meets the specification; the experimenter confirms `detected` against `review/<target>/answers.json`. Reported as detection rate on defective artifacts and false-alarm rate on correct ones |
| Self-report | `out/<test>/REPORT.md`: attempts, pitfalls, unclear points, tools used (F) |

`analyze.py` aggregates all of it into one table per (agent, target,
effort, condition) plus rows pooled over targets, over agents, and overall.

## Environment isolation

Every run gets its own sandbox directory and, with `--venv`, its own Python
environment built by uv from the condition's pinned requirements. The
sandbox contains the prompt (also as `CLAUDE.md` and `AGENTS.md` so agents
load it automatically), the task, a read-only `inputs/` symlink, an empty
`out/<test>/`, the condition's doc pack or allowed list, and a
`.claude/settings.json` that denies web access, package installs and reads
outside the sandbox. Nothing from `references/`, `solutions/`, `review/` or
`grade.py` is inside a sandbox. Each task tells the participant this in its
"Environment" section.

## Layout

| Path | What |
|---|---|
| `targets.py` | Facts about the three surgery targets; the single source for everything generated |
| `generate.py` | Renders `tasks/*/TASK-<target>.md`, `solutions/<target>/{P,B}/`, `review/<target>/` |
| `tasks/<test>/TASK-<target>.md` | Specification given to the participant |
| `conditions/{P,F,B}.md` | Condition preamble prepended to the task |
| `conditions/requirements-*.txt`, `constraints-B.txt`, `F-allowed.md`, `sandbox-settings.json` | Environment contents and sandbox permissions |
| `record-template.md` | Every recorded field, error classes, review record, participant self-report |
| `setup.py` | Builds inputs and hidden references for every target |
| `make_docpack.py` | Assembles `docpack/` for condition B (README, interfaces reference, full `help` dump, two frozen example plans) |
| `make_sandbox.py` | Creates one run directory (and optionally its environment) |
| `run_claude.py`, `run_codex.py` | Drive one participant session with Claude Code or Codex CLI: sandbox, solve, grade, review |
| `run_matrix.py`, `run_matrix_codex.py` | Run a Claude or Codex matrix resumably and in parallel |
| `run_full_claude.sh`, `run_full_codex.sh` | Run a full vendor matrix; the Codex launcher handles one repeat at a time on Linux |
| `audit_codex.py` | Recompute Codex harness fields from transcripts and report manual bookkeeping still required |
| `resummarise.py` | Recomputes execution counts in `harness.json` from saved Claude transcripts |
| `grade.py` | Grades an output against `references/<target>/<test>`; independent of BrainSurgery |
| `analyze.py` | Aggregates run records into the study tables |
| `compare_artifacts.py` | Lays the implementations of one test side by side across conditions, agents and tiers, with size and execution counts |
| `make_manifest.py`, `manifest.sha256` | Checksums of every input, reference and doc-pack file; `--verify` proves another machine runs the same study |
| `pack_data.sh` | Bundles the generated inputs and references (~35 GB) for transfer to another machine |
| `solutions/<target>/P/*.py`, `solutions/<target>/B/*.yaml` | Reference baselines and plans. Hidden from participants. The Python ones generate the references |
| `review/<target>/` | Defective variants of the references, one injected defect per test, plus `answers.json` |
| `AGENTS.md`, `CLAUDE.md` | Operating procedure for the experimenter agent |
| `inputs/<target>`, `references/<target>` | Symlinks into the data root (gitignored) |

## Setup

```bash
cd /path/to/brainsurgery
.venv/bin/python -c "import torch, safetensors, brainsurgery"
# base checkpoints (once): models/gpt2, models/olmo-1b-0724-hf, models/pythia-1b
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
# pinned revisions (also in targets.py); a different revision changes every derived file
snapshot_download("openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
                  local_dir="models/gpt2", allow_patterns=["*.json", "*.safetensors", "*.txt"])
snapshot_download("allenai/OLMo-1B-0724-hf", revision="d7cbab742d80589e714b1a2d7f838dcd21cbe143",
                  local_dir="models/olmo-1b-0724-hf", allow_patterns=["*.json", "*.safetensors", "*.txt"])
snapshot_download("EleutherAI/pythia-1b", revision="f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
                  local_dir="models/pythia-1b", allow_patterns=["*.json", "*.safetensors", "*.txt"])
PY
.venv/bin/python usability_tests/generate.py     # only after editing targets.py or generate.py
.venv/bin/python usability_tests/setup.py        # ~5 min, writes ~48 GB under models/usability_tests
.venv/bin/python usability_tests/make_docpack.py
```

`setup.py` writes `inputs/<target>/{base,ft1,ft2,lora}` and
`references/<target>/T1..T5` under `--data-root` (default
`models/usability_tests`). The fine-tunes are synthetic frozen-backbone
fine-tunes (seeded low-rank deltas on the MLP weights, small noise on MLP
biases, everything else bit-identical); the adapters are seeded PEFT-style
LoRA factors with an `adapter_config.json`. References come from the Python
baselines only, so grading is independent of the tool under test.

## Verified Linux handoff

Do not regenerate the seeded inputs on the receiving machine: QR and matrix
operations are not bit-identical across CPU/BLAS implementations. Transfer the
root-level `usability_tests_data.tar` separately from Git. The current bundle
is 50,568,509,440 bytes with SHA-256
`7001fedf486d026eea60213108278e4ddba41de40cb5b4a6d0db28f4dbcd28fe`.
Its tracked verification record and the complete receiving-machine sequence
are in `revision_tests/plans/linux_handoff_manifest.json` and
`revision_tests/plans/linux_handoff.md`.

The active data root is `models/usability_tests` (underscore). Do not use or
transfer the obsolete local `models/usability-tests` directory (hyphen). With
the three base checkpoints already downloaded at the revisions in
`targets.py`, the receiving sequence is:

```bash
echo "7001fedf486d026eea60213108278e4ddba41de40cb5b4a6d0db28f4dbcd28fe  usability_tests_data.tar" | sha256sum -c -
tar -xf usability_tests_data.tar -C models/
.venv/bin/python usability_tests/setup.py
.venv/bin/python usability_tests/make_docpack.py
.venv/bin/python usability_tests/make_manifest.py --verify
```

Stop unless the last command says `verified 126/126 files`. The official
Codex namespace is `astra_eacl2027`; `astra` is reserved for the two excluded
macOS pilot cells and an incomplete pilot sandbox already tracked in the kit.

## Grading

`grade.py <test> --target <target> [--out PATH] [--json] [--write FILE]`
checks, in order: loadability (single `.safetensors`, torch `.pt`, or a
sharded directory with an index); the sharding rule for T3 and T5, where a
shard may exceed the budget only if it holds a single tensor; exact key set;
per-tensor shape and dtype; values. Values are bit-exact except for tensors
produced by floating-point arithmetic (the merged MLP tensors of T4 and the
merged attention weights of T5), which use a relative Frobenius tolerance of
1e-5 for float32 outputs and 1e-3 for half-precision outputs.

## Verification

Verified 2026-09-05 on this machine: every plan in `solutions/<target>/B`
passes `grade.py` against the Python-generated references under the default
provider, and every defective artifact in `review/<target>/` (Python and plan)
fails it. Wall clock is one BrainSurgery run on CPU; line counts are
non-blank, non-comment lines (`validation/count_lines.py`), the Python column
excluding the 45-line shared loader/writer `_ckpt.py`.

| Test | gpt-2 Python / plan / s | olmo-1b Python / plan / s | pythia-1b Python / plan / s |
|---|---|---|---|
| T1 | 27 / 16 / 4 | 27 / 19 / 5 | 27 / 19 / 4 |
| T2 | 32 / 60 / 4 | 34 / 83 / 6 | 32 / 64 / 5 |
| T3 | 20 / 14 / 4 | 20 / 13 / 4 | 20 / 14 / 5 |
| T4 | 28 / 20 / 4 | 28 / 20 / 10 | 28 / 24 / 7 |
| T5 | 31 / 20 / 4 | 31 / 19 / 4 | 31 / 21 / 4 |

T2 is much longer as a plan than as a script because `concat` takes single
tensor references only, so every per-layer concatenation is spelled out (16
layers times up to four tensors for OLMo); the renumbering moves in T1 are
spelled out for the same reason. Both are genuine usability findings about
the DSL and stay in the task set.

## Results

Four agents, two complete repeats, 1080 cells: Sonnet 5, Opus 5 and Fable 5.1
through Claude Code (`low`/`medium`/`high` tiers) and gpt-5.6-sol through the
Codex CLI (`light`/`medium`/`high`, namespace `sol_eacl2027`). Every agent ran
3 targets x 5 tests x 3 conditions x 3 tiers x 2 repeats = 270 cells. Records
are under `usability_tests/<agent>/<target>/<effort>/`; `analyze.py`
reproduces every table below.

Pooled over agents, targets and tiers:

| Condition | Success | First run OK | Median cost (USD) | Median time to solution | Median output tokens |
|---|---|---|---|---|---|
| P (Python) | 360/360 | 349/360 (97%) | 0.32 | 72 s | 3.5k |
| F (free choice) | 360/360 | 329/360 (91%) | 0.41 | 107 s | 4.9k |
| B (BrainSurgery) | 360/360 | 333/360 (92%) | 0.68 | 140 s | 6.1k |

By agent (pooled over targets and tiers), as P / F / B:

| Agent | First run OK | Median cost | Median time | Median output tokens |
|---|---|---|---|---|
| Sonnet 5 | 96% / 89% / 98% | 0.16 / 0.21 / 0.49 | 61 / 85 / 164 s | 4.0k / 5.7k / 11.2k |
| Opus 5 | 99% / 92% / 100% | 0.38 / 0.54 / 0.76 | 71 / 111 / 127 s | 4.4k / 6.8k / 7.1k |
| Fable 5.1 | 94% / 96% / 100% | 0.53 / 0.63 / 1.02 | 55 / 76 / 104 s | 3.1k / 4.1k / 5.3k |
| gpt-5.6-sol | 99% / 89% / 72% | 0.28 / 0.34 / 0.50 | 124 / 169 / 173 s | 2.8k / 4.1k / 3.7k |

By effort tier, Claude agents only (the Codex tiers are named differently and
map to reasoning effort, so they are not pooled with these):

| Tier | First run OK | Median cost | Median time |
|---|---|---|---|
| low | 94% / 93% / 99% | 0.30 / 0.36 / 0.61 | 50 / 65 / 97 s |
| medium | 99% / 91% / 99% | 0.37 / 0.53 / 0.78 | 59 / 84 / 115 s |
| high | 96% / 92% / 100% | 0.53 / 0.71 / 1.12 | 83 / 130 / 181 s |

By target (pooled over agents and tiers):

| Target | First run OK | Median cost | Median time |
|---|---|---|---|
| gpt-2 | 99% / 93% / 94% | 0.30 / 0.38 / 0.65 | 61 / 100 / 131 s |
| olmo-1b | 98% / 91% / 93% | 0.33 / 0.44 / 0.69 | 80 / 116 / 153 s |
| pythia-1b | 94% / 90% / 90% | 0.33 / 0.39 / 0.71 | 71 / 103 / 145 s |

### Correctness does not separate the conditions

All 1080 cells passed the hidden-reference grader, on every target, in every
condition, for every agent and tier, with no cap hits. For agents of this
capability the three ways of expressing checkpoint surgery are equally
reliable at producing a correct result. What separates them is effort, and the
ordering is the same everywhere: a BrainSurgery plan costs about twice a
Python script and takes about twice as long, with the free-choice condition in
between. Raising the effort tier scales cost and time in all three conditions
without changing any outcome.

First-run success is the one place the plan condition wins outright for the
Claude agents: 268 of 270, against 260 for Python and 249 for free choice.
That is what the executable assertions in a plan are for, and it is the
clearest quantitative support for the paper's inspectability claim. It does
not hold for the Codex agent, which is at 72% in condition B against 99% in
Python. Its 22 condition-B first-run failures are plan-authoring errors caught
by the plan itself before anything is written: schema rejections from the
loader, transform errors such as a path pattern matching zero tensors, and
failed `assert` and write-count checks. It recovered from every one of them
within the turn budget, and all 90 of its condition-B cells passed. The
comparison to draw is not that plans are harder for this agent, but that in
condition B a mistake surfaces as a refusal to run, where the Python condition
offers no equivalent check.

### Bug detection is saturated; false alarms are not

Odd repeats show the reviewer a defective artifact, even repeats the correct
reference, so each agent contributes 135 detection trials and 135 false-alarm
trials.

| Agent | Bug detected | False alarms | P / F / B false alarms |
|---|---|---|---|
| Opus 5 | 135/135 (100%) | 0/135 (0.0%) | 0 / 0 / 0 |
| Fable 5.1 | 135/135 (100%) | 0/135 (0.0%) | 0 / 0 / 0 |
| Sonnet 5 | 135/135 (100%) | 24/135 (17.8%) | 8 / 6 / 10 |
| gpt-5.6-sol | 135/135 (100%) | 32/135 (23.7%) | 10 / 11 / 11 |

Detection is 100% for every agent in every condition, so that half of the
review measure does not discriminate: the seeded defects are too easy for
frontier agents, and separating the conditions on detection would need harder
ones. The false-alarm column does discriminate, but it separates *models*, not
conditions. The two strongest models raise no false alarms at all; the two
others raise them at similar rates across P, F and B. Read the review phase as
a property of the reviewer rather than of the artifact format.

Total study cost is 707.35 USD: 575.28 for the solve phases and 132.07 for the
reviews. The Codex figures are imputed from the published rate card, because
the runs went through a ChatGPT subscription that reports no per-call cost;
they are not directly comparable to the metered Claude figures.

### Caveats

- Coding agents are proxies for practitioners, not substitutes. Every number
  here is agent latency and agent token cost.
- The tests were sized so that a competent agent can finish them. A ceiling at
  100% success means the tasks do not discriminate, not that the conditions
  are interchangeable at every difficulty.
- Codex has no turn cap, only the 30-minute time cap, so `cap_hit` is not
  comparable between vendors. No cell hit either cap.
- In 7 of the 56 false alarms the verdict text concedes, after its opening
  `NO`, that the behaviour it objects to is in fact correct ("this is actually
  consistent with the spec", "matches the spec's mapping"). All 7 are Sonnet 5
  and they are spread over P, F and B. They are scored on the first word, the
  pre-registered rule; the count comes from a phrase search, so read it as a
  floor rather than an exact figure.

## Doc-consultation time

Over both repeats (810 runs), `doc_time.py` measures how much of each solve
went into reading documentation: for every tool call that fetches
documentation, the span from the call to the first assistant message after
its result, that is the fetch plus the model turn that consumes it,
overlapping spans merged. The numbers below are condition B, the only
condition with a doc pack.

| Group | Runs | Median doc time | Median solve time | Share of solve | Median reads | Median KB read |
|---|---|---|---|---|---|---|
| all B | 270 | 35 s | 128 s | 28% | 7 | 49 |
| Sonnet 5 | 90 | 58 s | 164 s | 33% | 9 | 60 |
| Opus 5 | 90 | 28 s | 127 s | 22% | 7 | 42 |
| Fable 5.1 | 90 | 30 s | 104 s | 29% | 6 | 49 |
| low | 90 | 21 s | 97 s | 23% | 5 | 35 |
| medium | 90 | 34 s | 115 s | 29% | 7 | 47 |
| high | 90 | 54 s | 181 s | 30% | 8 | 65 |
| T1 | 54 | 29 s | 100 s | 27% | 6 | 40 |
| T2 | 54 | 32 s | 115 s | 28% | 6 | 48 |
| T3 | 54 | 23 s | 95 s | 24% | 6 | 39 |
| T4 | 54 | 65 s | 194 s | 33% | 8 | 65 |
| T5 | 54 | 40 s | 152 s | 26% | 8 | 60 |

Pooled, 200 of the 670 minutes of condition-B solve wall clock is doc
consultation, 30 percent; per run it ranges from 6 s to 296 s and is stable
across repeats (median 35 s and 36 s) and across targets (28 to 39 s). It
grows with the effort tier in every agent (Sonnet 5 34/50/67 s, Opus 5
18/27/46 s, Fable 5.1 18/28/52 s) and it is largest on T4, the multi-input
task-vector merge, which is also the longest task.

Two things bound how far this can be read:

- It is agent latency, not human reading time. It compares conditions,
  agents, tiers and tasks with each other; it is not an estimate of what a
  practitioner would spend on the same documentation.
- There is no symmetric baseline. P consulted nothing at all (0 of 270 runs),
  F inspected an installed package in 32 of 270 runs for 11 minutes in total;
  both work from what the model already knows about torch, peft and the merge
  toolkits. So the doc-reading cost is the cost of a tool the models have not
  memorised, and it cannot be phrased as a comparison against the
  documentation of the alternatives.

The attribution itself is tight: no assistant message ever mixed a doc read
with other tool calls, and only 41 of 1968 doc reads are followed by a
message that acts rather than thinks or reads on, so the upper and lower
bounds of the measure differ by 1 percent (200 min against 198 min).

## Pilot, and what it changed

A pilot (Sonnet 5, medium effort, one repeat, 45 cells) ran before the study
and is preserved in git history (commit 603b806 on the `usability-study`
branch); it is not part of the study data. It forced four changes, all made
before the study started:

- three documentation gaps that cost the plan condition most of its extra
  turns were fixed in the BrainSurgery README, the interfaces reference and
  the `assert equal` help text: shard sizes are binary units and count tensor
  data only (an oversized tensor goes alone in its shard); which alias a
  multi-input plan writes as output; and that `assert equal`'s `right` is a
  rewrite of each `left` match, so capture groups work across aliases;
- condition B installs BrainSurgery non-editable so the repository source is
  not reachable from the sandbox;
- inputs and base checkpoints are read-only (a participant wrote through a
  copied symlink into the shared GPT-2 base);
- execution counting matches real invocations only.

The doc pack is regenerated from the fixed documentation; every study cell
sees the same pack.

## Fixed on the way

The in-memory provider used to fail at save time on any output that still
held a non-contiguous tensor (a `permute` result or a `phlora` factor),
after all transforms had succeeded, leaving partial shards on disk. Fixed
2026-09-05 in `brainsurgery/io/safetensors.py` (tensors are packed at save
time), regression test in `tests/test_io.py`, recorded in `wiki/log.md`.
Output is still not atomic on other save-time failures.

## Open items

1. The seeded defects are detected 100% of the time by every agent. Separating
   the conditions on review quality needs harder defects, which would mean a
   second generation of `review/` artifacts and a fresh review pass.
2. Doc-consultation time exists for the Claude cohort only. `doc_time.py`
   reads the Claude Code stream-json transcript; a Codex column needs a reader
   for the `codex exec --json` event stream.
3. Codex cost is imputed from the rate card, because the runs went through a
   ChatGPT subscription that reports no per-call cost. Metered runs would make
   the cost column comparable across vendors.
4. The sandbox permission denies cover Claude Code participants; other
   agents need the equivalent in their own driver, or a container.
