# T4 — Participant self-report (condition F)

- **Final artifact path:** `out/T4/solution.py` (entry point `out/T4/run.sh`);
  output written to `out/T4/model.safetensors`.

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none — the first
  execution succeeded and all checks passed.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious ordering hazard: both task vectors must be taken against the
    *unmodified* base, so I bind `bf = base[X]` once per tensor and derive both
    `tv1` and `tv2` from it, rather than accumulating into a running result.
  - The base is a two-shard directory while `ft1`/`ft2` are single files, so the
    loader has to resolve `model.safetensors.index.json` for one input and a
    plain `safe_open` for the other two.
  - "Identical" needs to be bit-exact, and `torch.equal` reports NaN != NaN;
    I compare the raw bytes (`view(torch.uint8)`) instead so the verification
    means the same thing for every bit pattern.
  - Checkpoints are ~5 GB each and there are three of them, so tensors are
    pulled one name at a time through `safe_open` instead of loading three full
    state dicts; the output dict is freed before the re-read verification pass.
  - I deliberately did *not* assert "all 48 MLP tensors differ from base" as a
    pass condition — a fine-tune could legitimately leave one tensor untouched.
    The required "exactly 48 merged" check is enforced on the merge loop itself.

- **Anything in the task text or documentation that was unclear:** nothing
  material. One small ambiguity: the task says the output must be "a single
  file with exactly 114 tensors" but does not say whether HF sidecar files
  (`config.json`, tokenizer) should be copied into `out/T4/`; since grading
  compares tensors only, I wrote just the safetensors file (plus my own
  artifacts). I also assumed safetensors `__metadata__` is not graded and set
  it to `{"format": "pt"}` for HuggingFace compatibility.

- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — lazy per-tensor reads (`safe_open`) of the sharded
    base and the two fine-tune files, and `save_file` for the output. Chosen
    because it lets me stream 15 GB of input without materialising three state
    dicts, and because it preserves names, shapes and dtypes exactly.
  - `torch` 2.14.0 — float32 tensor arithmetic for the task-vector merge and
    bit-exact comparison for the shared-tensor verification.
  - `numpy` 2.5.2, `transformers` 5.12.1 — available but not used.
  - **`mergekit` 0.1.4 — considered and rejected.** Its `task_arithmetic`
    method is the nominal route for this task, but it applies the merge to
    *every* tensor it finds rather than to a named 48-tensor subset, it has no
    way to express the precondition "abort unless all 66 non-MLP tensors are
    bit-identical across the three checkpoints", and it re-serialises through a
    HF model with its own sharding and dtype policy rather than emitting one
    file with exactly the input's 114 names. Every one of the three required
    checks would have had to live in a wrapper script anyway, at which point the
    wrapper is the whole solution and mergekit only adds a failure surface.
  - **`peft` / `torch-state-bridge` — not applicable:** no adapters here, and
    no key renaming is required (names must not change).

  So: a ~180-line script directly on `safetensors` + `torch`. The task is
  arithmetic plus verification, and the verification is the part no off-the-shelf
  merge tool expresses.

- **Approximate time spent, if you can tell:** roughly 10 minutes end to end;
  the merge itself runs in ~19 s wall clock.

## What the run enforces

All checks are hard failures (`CheckError` → exit 1), never warnings:

1. **Shared-tensor verification (before any arithmetic):** the three
   checkpoints have identical name sets; the set has 114 names; the MLP subset
   is exactly the 3 projections × 16 layers; each of the 66 non-MLP tensors
   matches in shape, dtype and every bit across base, ft1 and ft2.
2. **Exactly 48 merged:** the merge loop counts, and asserts `merged == 48`;
   the on-disk pass independently re-verifies 48 merged values and asserts the
   count again.
3. **Exactly 114 tensors out:** asserted on the in-memory dict, on its key set
   versus the base, and again after re-opening the written file.

Additionally, the written file is read back and checked tensor by tensor: every
non-MLP tensor must be bit-identical to the base, and every MLP tensor must
match an independently recomputed `base + 0.4*(ft1-base) + 0.4*(ft2-base)` to a
relative Frobenius error of 1e-7 (100× tighter than the 1e-5 grading bound).

Observed run output:

```
[plan] 48 MLP tensors over layers 0..15, 66 shared tensors, lambda=0.4
[check] 66 non-MLP tensors verified identical in base, ft1 and ft2
[write] out/T4/model.safetensors (5119161112 bytes)
[check] on disk: 114 tensors, 48 merged values re-verified (48 differ from base), 66 shared tensors bit-identical to base
OK
```
