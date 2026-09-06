# Participant self-report: T2 (GPT-2 124M, condition P)

- Final artifact path: `out/T2/solution.py` (output checkpoint: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Conv1D `[in, out]` layout means heads are column blocks of `c_attn` and row blocks of `c_proj`; the task text spelled this out, so no discovery was needed.
  - Built one 704-entry keep index per 768-wide segment and offset it by 0/768/1536 for the fused q|k|v columns, which avoids hand-typing six ranges.
  - Used `index_select` plus `.contiguous()` so safetensors never sees a view or shared storage.
- Anything in the task text or documentation that was unclear: nothing; the explicit column and row ranges removed any ambiguity about the head-5 boundaries.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3 only).
- Approximate time spent, if you can tell: about 2 minutes.
