# T3 participant self-report

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The 256 MiB budget is exactly 8 x 32 MiB bf16 [8192,2048] matrices, so shard boundaries must be computed on tensor data bytes, not file size, to stay at the limit without going over.
  - Embedding and lm_head (412 MB each, float32) exceed the budget and had to be placed alone in their own shards.
- Anything in the task text or documentation that was unclear: nothing significant; the task fully specifies the target set, so an anchored regex over the seven projection names was sufficient.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: dtype cast with `tensor.to(torch.bfloat16)` (RNE) as required.
  - `safetensors` 0.5.3: `load_file`/`save_file` for reading the two input shards and writing the output shards.
  - Did not use `transformers.save_pretrained`: it applies one dtype to the whole model and its sharding is by file size, so mixed precision plus the exact 256 MiB data budget was simpler to enforce in a plain script.
- Approximate time spent, if you can tell: about 2 minutes.
