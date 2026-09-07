# T3 participant self-report

- Final artifact path: `out/T3/solution.py` (output shards + `model.safetensors.index.json` in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Projection matrices had to be matched with an anchored regex so `.weight` did not also hit embeddings, layer norms or the `attention.bias` mask buffer.
  - Embedding matrices in float32 (412 MB each) exceed the 256 MiB shard budget, so the sharder must place them alone rather than fail.
- Anything in the task text or documentation that was unclear: the task says the embeddings are 206 MB each, which is their float16 size; after the required upcast they are 412 MB, still handled by the "alone in its own shard" rule.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: dtype casts (`to(torch.bfloat16)` / `to(torch.float32)`).
  - `safetensors` 0.5.3: `safe_open` to read, `save_file` per shard to write; index JSON written by hand.
  - Plain script rather than `transformers.save_pretrained`: that route would re-register the dropped buffers and cannot express per-tensor mixed dtypes without extra post-processing.
- Approximate time spent, if you can tell: ~2 minutes
