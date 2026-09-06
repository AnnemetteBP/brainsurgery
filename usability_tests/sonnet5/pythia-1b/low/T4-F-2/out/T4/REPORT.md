## Participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none — straightforward once the MLP key set was enumerated per-layer
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, used directly in a plain script rather than `mergekit`; the checkpoints are single small safetensors files with a simple task-arithmetic formula, so a script gives full control over the shared-tensor verification (name-set equality plus bit-exact identity of the 180 non-MLP tensors) and the exact-count checks the task requires, without needing mergekit's YAML merge-method abstraction.
- Approximate time spent, if you can tell: a few minutes
