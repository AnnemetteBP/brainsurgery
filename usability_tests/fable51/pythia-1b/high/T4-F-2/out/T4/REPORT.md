# T4 (Pythia-1B, condition F) — participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - None blocking. Loaded the three 2 GB checkpoints lazily via `safe_open` so the precondition pass and the merge never hold more than one tensor per checkpoint in memory at a time.
  - Compared unchanged tensors on their raw int16 bit patterns rather than float values, so a NaN in the base would not spuriously fail (or pass) the equality check.
  - Guarded against the destination already existing and re-opened the written file to confirm key set, shapes, dtypes and bit-exactness of the 180 untouched tensors.
- Anything in the task text or documentation that was unclear: nothing material. The task lists the MLP tensor names explicitly, which made the 64-tensor selection unambiguous (anchored regex over layer index 0..15 and the two `dense_*` names, verified to hit exactly 64).
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: lazy tensor-by-tensor reads (`safe_open`) and single-file write (`save_file`).
  - `torch` 2.14.0: float32 arithmetic for the task vectors, float16 cast, bit-exact equality checks.
  - Not used: `mergekit` 0.1.4. Its `task_arithmetic` method computes the same formula (base + Σ w_i·(ft_i − base)), but it does not perform the required shared-tensor precondition check, writes a sharded HF directory plus config files rather than a single `model.safetensors`, and would still need a wrapper script for the "exactly 64 merged / 244 total" checks. A ~100-line script on safetensors+torch enforces every required check directly and keeps the ordering hazard (each task vector taken against the unmodified base) explicit.
- Approximate time spent, if you can tell: about 3 minutes wall clock, including one ~11 s execution and a separate float64 spot-check (relative Frobenius error ≈ 2e-4 on sampled merged tensors, embeddings bit-identical to base).
