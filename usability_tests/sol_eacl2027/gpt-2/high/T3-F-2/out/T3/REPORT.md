# Participant self-report

- Final artifact path: `out/T3/model.safetensors.index.json` and `out/T3/model-00001-of-00004.safetensors` through `out/T3/model-00004-of-00004.safetensors` (reproducer: `out/T3/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None; the first execution succeeded.
- Pitfalls or surprises you hit (one line each): The 64 MiB limit applies to summed tensor payload bytes rather than safetensors file size, and `wte.weight` must be isolated because it individually exceeds the limit.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): PyTorch 2.14.0 for exact float32-to-bfloat16 conversion and tensor checks; safetensors 0.5.3 for checkpoint loading, shard writing, and read-back validation; Python standard library for deterministic sharding and index JSON generation.
- Approximate time spent, if you can tell: About 5 minutes.
