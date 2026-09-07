# T2 self-report (condition P, Pythia-1B)

- Final artifact path: `out/T2/model.safetensors` (script: `out/T2/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - The fused `query_key_value` layout is GPT-NeoX interleaved (per-head `[q|k|v]` blocks of 768 rows), not `[q | k | v]` segments; slicing it as three segments would silently produce a loadable but wrong checkpoint.
  - Head 5 lives on different axes in different tensors: rows 3840..4607 of `query_key_value.{weight,bias}` but columns 1280..1535 of `attention.dense.weight`.
  - `attention.dense.bias` is per output feature, not per head, so it must stay `[2048]`; the attention buffers likewise stay untouched.
  - `index_select` results were forced `.contiguous()` before `save_file` to avoid safetensors complaining about views into the source storage.
  - Kept the input file's safetensors metadata so the output header matches the reference.
- Anything in the task text or documentation that was unclear: nothing; the explicit row/column ranges in "Required result" removed the ambiguity about the interleaved layout.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: a few minutes; one read of the task, one script, one run.
