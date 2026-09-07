# T4 run record (condition F, Pythia-1B)

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  - Execution 1 — `crash`: `RuntimeError: self.dim() cannot be 0 to view Half as Byte`; my
    bit-exact comparison helper did `t.view(torch.uint8)` and the checkpoint contains a
    0-dim tensor, which cannot be re-viewed at a different element size. Fixed by
    flattening with `.reshape(-1)` before the byte view.
- **Pitfalls or surprises you hit (one line each):**
  - A 0-dim scalar tensor among the 244 keys broke the naive `view(torch.uint8)` bit compare.
  - Bit-exactness, not `allclose`, is what step 1 and the grader's "180 bit-exact" ask for,
    so the comparison must go through the raw byte pattern (also correct for NaN / -0.0).
  - The ordering hazard is easy to sidestep by never mutating the base: both task vectors are
    formed from the same `b32` float32 copy, so no in-place accumulation can contaminate them.
  - I derived the 64 MLP names from the layer range rather than by substring-matching `mlp`,
    so an unexpected extra `mlp.*` tensor would be caught by the count check instead of
    being silently merged.
- **Anything in the task text or documentation that was unclear:** Nothing blocking. Step 1
  says "identical", which I read as bit-identical (consistent with the grader's bit-exact
  check on the 180 unchanged tensors) rather than within a tolerance.
- **Tools used (condition F):**
  - `torch` 2.14.0+cu130 — float32 arithmetic and the bit-exact tensor comparison.
  - `safetensors` 0.5.3 — lazy per-tensor reads via `safe_open` (three 2 GB checkpoints are
    never fully resident at once) and `save_file` for the output.
  - **Not used: `mergekit` 0.1.4.** Its `task_arithmetic` method computes exactly this
    formula and would have been the obvious route, but the task's hard requirement is that
    the run *aborts* if the frozen-backbone precondition does not hold, and if exactly 64
    tensors are merged. A mergekit YAML has no place to express either check: it would
    happily merge all 244 tensors and produce a plausible-looking checkpoint whether or not
    the backbone was actually shared. Wrapping mergekit in a script that pre-verifies the
    precondition and post-verifies the counts would have meant writing the same checks
    anyway, plus reconciling mergekit's sharded/`save_pretrained` output with the required
    single `model.safetensors`. ~90 lines of torch + safetensors was the smaller, more
    auditable artifact.
- **Approximate time spent, if you can tell:** ~5 minutes; the merge itself runs in ~9 s.

## Checks enforced by the run

All raise `CheckFailed` and exit non-zero:

1. `base`, `ft1`, `ft2` have identical key sets, of size 244; the 64 expected MLP names are present.
2. All 180 non-MLP tensors are bit-identical (and same shape/dtype) in all three checkpoints.
3. Shapes and dtypes of each merged tensor agree across the three checkpoints.
4. Exactly 64 tensors were merged; the in-memory output dict has exactly 244 entries.
5. Re-read from disk: the written file has 244 tensors, the base's key set, matching
   shapes/dtypes, and no tensor outside the 64 MLP names differs from the base.

Observed on the passing run: 180 identical non-MLP tensors, 64 merged at lambda=0.4,
244 tensors written, 180 bit-identical to the base. Independent spot check of
`gpt_neox.layers.7.mlp.dense_h_to_4h.weight` against a freshly recomputed float32
reference: relative Frobenius error 2.0e-4 (fp16 rounding), tolerance 1e-3.
