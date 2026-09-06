# T3 self-report

- Final artifact path: `out/T3/plan.yaml` (output written to `out/T3/`, 9 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all asserts and wrote the sharded output.
- Pitfalls or surprises you hit (one line each):
  - Ordering matters: delete the buffers first, then `cast_ '.*' -> float32`, then narrow the 64 projections to bfloat16; a broad `.*weight` cast would have hit embeddings and norms.
  - The uint8 causal-mask buffers would have been cast by the blanket float32 step, so they had to be deleted before it.
  - Shard budget: the README states `MB` is binary, so `256MB` is exactly the 268,435,456-byte budget; in float32 the embedding matrices are 412 MB each and were auto-placed alone in their own shards.
- Anything in the task text or documentation that was unclear: the task quotes the embeddings as "206 MB each", which is their float16 input size; after the required float32 upcast they are 412 MB, still oversized so the rule applies either way.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~5 minutes.
