# T4 — Participant self-report (condition P)

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution
  succeeded and all checks passed.
- **Pitfalls or surprises you hit (one line each):**
  - GPT-2's 160 tensors include the 12 per-layer `h.<i>.attn.bias` causal-mask
    buffers, so equality checks on non-MLP tensors have to work for non-float
    dtypes too (used `torch.equal`, which also enforces shape and dtype).
  - `mlp.c_proj` vs `attn.c_proj`: the MLP regex is anchored on
    `h.<i>.mlp.c_(fc|proj).(weight|bias)$` so it cannot overreach onto the
    attention projections; the count is asserted to be exactly 48 and the
    matched layer indices to be exactly 0..11.
  - Ordering hazard: both task vectors are computed against tensors read
    straight from the base file inside a single expression, so the second
    vector can never be taken against an already-merged base.
  - Wrote each output tensor with `.clone().contiguous()` to avoid
    safetensors rejecting shared/aliased storage (GPT-2 ties `wte.weight`).
- **Anything in the task text or documentation that was unclear:**
  - The task lists `h.<i>.mlp.c_fc.weight` as `[768, 3072]`, i.e. Conv1D
    `[in, out]` layout rather than Linear `[out, in]`; irrelevant for an
    elementwise merge, but worth noting since it is the opposite of the
    usual convention. The regex tolerates an optional module prefix
    (e.g. `transformer.`) in case the checkpoint carried one.
- **Tools used (condition F):** n/a — condition P: `torch` 2.14.0,
  `safetensors` 0.5.3 only.
- **Approximate time spent, if you can tell:** ~5 minutes.

## What the script does

1. Opens the three checkpoints lazily with `safe_open`, asserts the key sets
   are identical and that each holds 160 tensors.
2. Selects the MLP tensors by anchored regex, asserts exactly 48 across
   layers 0..11.
3. Per key: asserts equal shape and dtype across all three; for non-MLP keys
   asserts `torch.equal(base, ft1)` and `torch.equal(base, ft2)` and aborts
   otherwise; for MLP keys asserts float32 and computes
   `base + 0.4*(ft1-base) + 0.4*(ft2-base)`.
4. Asserts 48 merged / 112 copied / 160 total before saving.
5. Reopens the written file and re-verifies: 160 keys, same key set, shapes
   and dtypes match the base, and the 112 unchanged tensors are bit-identical
   to the base.
