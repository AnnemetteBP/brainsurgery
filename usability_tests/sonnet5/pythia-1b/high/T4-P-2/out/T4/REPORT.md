# Participant self-report

- Final artifact path: `out/T4/solution.py` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script: 2
- Which executions failed, and why (one line each):
  - Execution 1: `crash` — `FileNotFoundError` on `inputs/base/model.safetensors`; the script computed the sandbox root from `solution.py`'s location with the wrong number of `.parents[]` hops, pointing one directory too high.
- Pitfalls or surprises you hit (one line each):
  - `inputs/` also contains a `lora/` directory not mentioned in TASK.md; ignored it since the spec only names `base`, `ft1`, `ft2`.
  - Needed to be careful to diff non-MLP tensors against *both* ft1 and ft2 independently (not just check ft1==ft2==base), so a defect unique to only one fine-tune outside the MLP set would still be caught.
- Anything in the task text or documentation that was unclear: None; the merge formula and required checks were unambiguous.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes.
