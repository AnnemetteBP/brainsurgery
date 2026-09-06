# T4 — Task-vector merge of two GPT-2 fine-tunes (condition F)

## Participant self-report

- **Final artifact path:** `out/T4/solution.py` (run as
  `.venv/bin/python out/T4/solution.py`); output `out/T4/model.safetensors`.
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The checkpoints use flat GPT-2 keys (`h.0.mlp.c_fc.weight`, `wte.weight`), not the
    HF `transformer.`-prefixed layout, so loading them as `GPT2LMHeadModel` and merging
    via a model-level toolkit would have needed key rewriting for no benefit.
  - The ordering hazard the task warns about: both task vectors must be taken against
    the original `base`, so I read `base` once into `b` and computed `b + λ·(ft1−b) + λ·(ft2−b)`
    in one expression rather than folding ft1 into the base first.
  - `h.<i>.attn.bias` is a non-float causal mask buffer; the shared-tensor check uses
    `torch.equal` (dtype/shape-aware) rather than a float tolerance, so it handles those.
  - Unchanged tensors are graded bit-exact, so I copy them straight from the base and
    never route them through float arithmetic.
- **Anything in the task text or documentation that was unclear:** nothing material.
  The formula, λ, the 48-tensor list and the output path were all explicit.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save the three checkpoints and the output. Direct
    state-dict access is exactly the granularity this task needs.
  - `torch` 2.14.0+cu130 — float32 tensor arithmetic and the `torch.equal` /
    `torch.allclose` comparisons.
  - Considered and rejected: **mergekit** 0.1.4 `task_arithmetic`. It is the advertised
    route for T4, but it merges *every* tensor of a model rather than a named 48-tensor
    subset, it offers no way to assert the frozen-backbone precondition (step 1) or to
    assert "exactly 48 merged", it wants HF model directories whose key layout does not
    match these files, and its normalization/dtype defaults would put the 112 unchanged
    tensors at risk of not being bit-exact. A ~100-line script over safetensors expresses
    the spec directly and makes every required check a hard failure.
  - `transformers` / `peft` were not needed; no adapter or architecture-level operation
    is involved.
- **Approximate time spent:** ~5 minutes.

## How the required checks are enforced

All checks raise `CheckFailed`, which exits with status 1 and writes nothing:

1. **Shared-tensor verification (step 1):** identical key sets across base/ft1/ft2
   (both missing and extra keys), base has 160 tensors, the 48 MLP tensors are present
   with the documented shapes and float32 dtype, and each of the 112 non-MLP tensors is
   `torch.equal` in all three checkpoints.
2. **Exactly 48 tensors merged:** the merge loop counts merged tensors and asserts `== 48`;
   the key list itself is asserted to have 48 unique entries.
3. **Output has exactly 160 tensors:** asserted before writing, and re-asserted after
   reloading the file from disk.

Additionally, a post-write pass reloads `out/T4/model.safetensors` and verifies every
merged tensor against the formula and every other tensor bit-identical to the base.

## Verification performed

- Independent re-check of the written file: 160 keys matching the base, 48 tensors differ
  from the base and 112 are bit-identical, worst relative Frobenius error vs the formula
  `0.00e+00`, all merged tensors `F32`.
- Negative control (on a scratch output path, not the graded file): dropping a key from
  ft1, adding an extra key to ft1, and perturbing one backbone tensor in ft2 by 1e-6 each
  aborted with the expected `CheckFailed`.
