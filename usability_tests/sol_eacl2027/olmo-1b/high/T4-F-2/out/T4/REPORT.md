# Participant self-report

- Final artifact path: `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The base is split across two indexed shards while each fine-tune is a single safetensors file, so the solution uses one checkpoint abstraction and validates the index against the shard contents.
- Pitfalls or surprises you hit (one line each): The merge must retain every MLP result until the single-file save, so input tensors are memory-mapped and tensors are processed individually to keep peak memory bounded.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): `safetensors` 0.5.3 for memory-mapped checkpoint reads and the required single-file output; `torch` 2.14.0 for exact equality checks and float32 task-vector arithmetic.
- Approximate time spent, if you can tell: About 5 minutes.
