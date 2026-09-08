## Participant self-report

- Final artifact path: `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The fused QKV tensor is interleaved by head, so head 5 is one contiguous 768-row block rather than three slices in separate Q/K/V regions.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): Not applicable (condition P).
- Approximate time spent, if you can tell: About 2 minutes.
