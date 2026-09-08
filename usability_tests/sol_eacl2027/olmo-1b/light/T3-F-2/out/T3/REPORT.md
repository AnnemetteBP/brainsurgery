## Participant self-report

- Final artifact path: `out/T3/model.safetensors.index.json` and the ten `out/T3/model-*.safetensors` shard files; producer: `out/T3/solution.py`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The two FP32 embedding/head tensors exceed the shard limit and therefore each must be isolated, while the BF16 projection shards can land exactly on the 256 MiB limit.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): Python with torch 2.14.0 for exact BF16/FP32 conversion, and safetensors 0.5.3 for bounded-memory tensor reads and sharded writes; chosen for direct, explicit control over tensor selection and shard packing.
- Approximate time spent, if you can tell: About 5 minutes.
