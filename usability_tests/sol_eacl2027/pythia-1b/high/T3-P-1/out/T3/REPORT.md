# Participant self-report

- Final artifact path: `out/T3/`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The two embeddings become about 393 MiB each after the required float32 upcast, so each must use the single-oversized-tensor shard exception.
- Anything in the task text or documentation that was unclear: The task describes the embeddings as 206 MB each, which matches their float16 input size; their required float32 output size is 412,090,368 bytes each.
- Tools used (condition F): N/A (condition P).
- Approximate time spent, if you can tell: About 5 minutes.
