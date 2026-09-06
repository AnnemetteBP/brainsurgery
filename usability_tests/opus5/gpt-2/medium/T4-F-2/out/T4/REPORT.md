# T4 — Participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - Ordering hazard called out in the task: both task vectors must be taken
    against the *unmodified* base, so I compute `v1` and `v2` from the loaded
    base tensor before writing anything into the output dict; nothing is
    accumulated in place.
  - GPT-2 uses Conv1D layout, so `c_fc.weight` is `[768, 3072]` and
    `c_proj.weight` is `[3072, 768]`; irrelevant here since the merge is
    elementwise, but it rules out any transpose "fix".
  - Non-float buffers exist among the 112 untouched tensors (e.g.
    `h.<i>.attn.bias` causal masks), so the shared-tensor check uses
    `torch.equal` (bit-exact, dtype-agnostic) rather than an allclose.
  - Untouched tensors are copied from the base and re-verified bit-exactly
    after the file is written, since grading requires bit-exactness there.
- **Anything in the task text or documentation that was unclear:** nothing
  material. The formula, lambda, key list and counts are all explicit; I
  assumed "computed in float32" means the accumulation dtype and the stored
  dtype both stay float32, which matches the inputs.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save of the three checkpoints and the output.
  - `torch` 2.14.0 — float32 arithmetic and the bit-exact `torch.equal`
    comparisons.
  - Considered and rejected: `mergekit` 0.1.4 task arithmetic. It expresses
    the same formula, but it merges *all* tensors rather than a named subset,
    it cannot express the required precondition ("every non-MLP tensor is
    identical in all three checkpoints, abort otherwise"), and it rewrites the
    output as a HuggingFace model directory with its own key naming and
    sharding rather than a single flat `model.safetensors` with these exact
    160 keys. Enforcing the required checks around it would have been more
    code than the 100-line script, with a bigger risk of silently touching the
    112 tensors that must stay bit-exact. `transformers`/`peft` do not apply:
    there are no adapters or architecture changes here.
- **Approximate time spent:** ~5 minutes.

## What the script enforces (fails loudly, exit code 1)

1. `set(base) == set(ft1) == set(ft2)` and the base has exactly 160 tensors.
2. All 48 expected MLP names are present; shapes and dtypes agree across the
   three checkpoints for every key.
3. All 112 non-MLP tensors are bit-exactly equal in base, ft1 and ft2.
4. Exactly 48 tensors were merged; the output dict has exactly 160 tensors
   with the base's key set.
5. After writing, the file is re-read: 160 tensors, unchanged shapes/dtypes,
   every non-MLP tensor still bit-exactly equal to the base, and at most 48
   tensors differing from the base.
