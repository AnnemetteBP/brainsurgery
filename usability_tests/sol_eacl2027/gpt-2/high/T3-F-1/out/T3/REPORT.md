# Participant self-report

- Final artifact path: `out/T3/model.safetensors.index.json` and `out/T3/model-00001-of-00004.safetensors` through `out/T3/model-00004-of-00004.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None; the first execution succeeded.
- Pitfalls or surprises you hit (one line each): The 64 MiB limit applies to tensor payload bytes, while the oversized `wte.weight` must be placed alone as an explicit exception.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): PyTorch 2.14.0+cu130 for the specified bfloat16 conversion; safetensors 0.5.3 for checkpoint loading, shard writing, and read-back verification; Python standard-library JSON for the index.
- Approximate time spent, if you can tell: About 5 minutes.
