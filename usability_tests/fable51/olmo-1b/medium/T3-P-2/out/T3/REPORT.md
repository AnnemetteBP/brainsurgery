# Participant self-report: T3 (olmo-1b, condition P)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 10 shard files plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Projection matrices are selected with an anchored regex on `model.layers.<i>.self_attn.{q,k,v,o}_proj.weight` and `mlp.{gate,up,down}_proj.weight`, so embeddings and `lm_head` are never touched.
  - The two 412 MB float32 tensors (`model.embed_tokens.weight`, `lm_head.weight`) exceed the 256 MiB shard budget and are stored alone in their own shards; the greedy packer flushes the current shard before and after them.
  - Each bfloat16 layer block is exactly 14 tensors = 256 MiB, so the packed shards land exactly on the budget (`<=` check, not `<`).
- Anything in the task text or documentation that was unclear: nothing; the task text lists the exact tensor names and the sharding rule explicitly.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
