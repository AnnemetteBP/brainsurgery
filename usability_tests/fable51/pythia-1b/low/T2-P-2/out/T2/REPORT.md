# Participant self-report: T2 (Pythia-1B head pruning, condition P)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion. A self-added check that every tensor is float16 fired; the attention buffers (masked_bias / bias mask) are not float16. The check was not required and was removed; no output was written.
- Pitfalls or surprises you hit (one line each):
  - Not every tensor in the "float16 checkpoint" is float16: the attention buffers have other dtypes.
- Anything in the task text or documentation that was unclear: nothing; the interleaved per-head qkv layout and kept row/column ranges were stated explicitly.
- Tools used (condition F): n/a (condition P: torch, safetensors).
- Approximate time spent, if you can tell: about 2 minutes.
