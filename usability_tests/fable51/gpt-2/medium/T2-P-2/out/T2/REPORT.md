# Participant self-report: T2 (GPT-2 124M, condition P)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Conv1D `[in, out]` layout means heads are columns in `c_attn` but rows in `c_proj`; the task text spelled this out, so no discovery needed.
  - Built the keep-index list per 768-wide segment (q, k, v) rather than hand-typing ranges, then cross-checked layer 0 against the explicit slices from TASK.md.
  - Made the sliced tensors contiguous before `save_file` to avoid safetensors rejecting non-contiguous views.
- Anything in the task text or documentation that was unclear: nothing; the explicit column/row ranges removed all ambiguity.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
