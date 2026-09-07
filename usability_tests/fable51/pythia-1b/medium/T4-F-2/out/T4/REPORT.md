# T4 participant self-report (condition F)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None. Inputs matched the task description exactly (244 names, 64 MLP tensors, float16, no diffs outside the MLP).
- Anything in the task text or documentation that was unclear:
  - Nothing material. The formula was stated explicitly enough that the ordering hazard (task vectors against the unmodified base) was easy to avoid by reading all three checkpoints into memory and never mutating the base dict.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: float32 arithmetic, `torch.equal` for the bit-exact shared-tensor check.
  - `safetensors` 0.5.3: `load_file` / `save_file` for checkpoint I/O.
  - mergekit (task_arithmetic) was considered and rejected: it does not enforce the precondition that all non-MLP tensors are identical across the three checkpoints, and it would require HF model loading and an extra config for something that is ~40 lines of plain torch. A script makes the required checks explicit and loud (`sys.exit` on failure).
- Approximate time spent, if you can tell: about 2 minutes, including an independent post-hoc verification (relative Frobenius error ~2e-4 on a sampled merged tensor, non-MLP tensors bit-identical to base, 64 MLP tensors changed).
