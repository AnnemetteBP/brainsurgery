# Participant self-report

- Final artifact path: `out/T5/solution.py`; generated checkpoint at `out/T5/model.safetensors.index.json` and `out/T5/model-00001-of-00005.safetensors` through `out/T5/model-00005-of-00005.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None; the first execution succeeded.
- Pitfalls or surprises you hit (one line each): The adapter product uses Linear `[out, in]` layout and therefore required transposition for GPT-2 Conv1D `[in, out]` weights.
- Pitfalls or surprises you hit (one line each): The 154,389,504-byte `wte.weight` exceeds the 100 MiB cap and had to occupy a legal single-tensor oversized shard.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): `torch` 2.14.0+cu130 for float32 matrix multiplication and tensor operations; `safetensors` 0.5.3 for direct checkpoint loading and sharded saving; Python standard-library `json` and `pathlib` for configuration, indexing, and paths. This direct route preserves untouched checkpoint tensors without instantiating GPT-2.
- Approximate time spent, if you can tell: About 5 minutes.
