# Participant self-report: T4 (GPT-2 124M), condition P

- Final artifact path: `out/T4/solution.py` (output `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None. Task vectors are computed against the original `base` dict, which is never mutated, so ordering is safe.
  - `inputs/lora/` exists in the sandbox but is not part of this task; it was ignored.
- Anything in the task text or documentation that was unclear: nothing; the formula, count of 48 MLP tensors and check list were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
