# T4 self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The ordering hazard is the whole task: both task vectors must be differences
    against the original base, so I read all three checkpoints up front and never
    write into `base` — the merged value goes into a separate `out` dict.
  - float16 inputs: I upcast to float32 for the arithmetic and cast back to the
    base dtype per tensor rather than assuming float16 globally.
  - `save_file` rejects shared/non-contiguous storage, so every output tensor is
    `.clone()`d (unchanged ones) or `.contiguous()` (merged ones).
  - I derived the 64 MLP names from the spec's pattern instead of regex-matching
    `mlp.` so that a name like an MLP layernorm could not be swept in by accident;
    the set size is asserted to be exactly 64 and asserted to exist in the base.
- **Anything unclear:** nothing blocking. The spec says "the base dtype" is
  float16; I read the dtype from each base tensor instead of hardcoding it, which
  is equivalent here. It was also unstated whether the fine-tunes must actually
  differ in all 64 MLP tensors — I did not require that, only that non-MLP
  tensors are identical (as specified). The output check confirmed all 64 do
  in fact differ.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save; the task is a pure state-dict rewrite and this
    is the direct API for it.
  - `torch` 2.14.0 — float32 arithmetic and `torch.equal` for the bit-exact
    shared-backbone verification.
  - I considered `mergekit` 0.1.4 `task_arithmetic`, which computes exactly this
    formula, but rejected it: it offers no way to enforce the required checks
    (bit-exact equality of the 180 non-MLP tensors, "exactly 64 tensors merged"),
    it merges every tensor rather than a named subset, and it would rewrite the
    checkpoint through a HF model round-trip that risks perturbing untouched
    tensors that must stay bit-exact. A ~90-line script enforces all three checks
    directly.
- **Approximate time spent:** ~4 minutes, one pass.

## Checks enforced by the run

All raise `CheckFailed` and exit non-zero:
1. identical key sets across base/ft1/ft2, and 244 keys in base;
2. matching shapes and dtypes for every tensor in all three;
3. all 64 expected MLP names present in base, and the name set is exactly 64;
4. every tensor outside those 64 is bit-identical in base, ft1 and ft2;
5. exactly 64 tensors merged; output holds exactly 244 tensors;
6. post-write re-read: 244 tensors, key set matches base, and no tensor outside
   the MLP set differs from base.
