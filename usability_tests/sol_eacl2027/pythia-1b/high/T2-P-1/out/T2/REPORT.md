# Participant self-report

- Final artifact path: `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The fused QKV tensor uses per-head 768-row blocks, while the dense output projection uses 256-column head blocks.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): Not applicable (condition P).
- Approximate time spent, if you can tell: About 3 minutes.
