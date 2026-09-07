# Participant self-report: T2 (GPT-2 124M, condition P)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Conv1D `[in, out]` layout means c_attn heads live on columns and c_proj heads on rows; the task text made this explicit.
  - Advanced indexing returns non-contiguous views in some cases, so `.contiguous()` was applied before `save_file`.
- Anything in the task text or documentation that was unclear: nothing; the explicit index ranges removed ambiguity.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
