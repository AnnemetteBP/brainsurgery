# Participant self-report

- Final artifact path: `out/T3/model.safetensors.index.json` and `out/T3/model-00001-of-00004.safetensors` through `out/T3/model-00004-of-00004.safetensors` (producer: `out/T3/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None; the first execution succeeded.
- Pitfalls or surprises you hit (one line each): The 154,389,504-byte `wte.weight` tensor exceeds the shard limit and therefore had to be emitted alone, while every multi-tensor shard had to remain at or below 67,108,864 tensor-data bytes.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): `torch` 2.14.0 for exact bfloat16 conversion and tensor inspection; `safetensors` 0.5.3 for checkpoint loading, shard writing, and independent reopening/validation; Python standard library for deterministic sharding and JSON index generation.
- Approximate time spent, if you can tell: About 4 minutes.
