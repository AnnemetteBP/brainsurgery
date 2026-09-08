# Participant self-report

- Final artifact path: `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): GPT-2's fused `c_attn` tensor required pruning head 5 separately from each of the q, k, and v segments before concatenating them in the original order.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): `safetensors` 0.5.3 for direct checkpoint loading and saving; `torch` 2.14.0 for exact tensor slicing and concatenation. A plain Python script was used because it makes the fused-QKV layout and required assertions explicit.
- Approximate time spent, if you can tell: About 3 minutes.

Authored solution: `out/T2/solution.py`.
