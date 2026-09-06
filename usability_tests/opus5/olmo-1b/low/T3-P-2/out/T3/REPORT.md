# T3 self-report (condition P)

- Final artifact path: `out/T3/solution.py` (output in `out/T3/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Had to anchor the projection regex to `^model\.layers\.\d+\....\.weight$` so it cannot reach `model.embed_tokens.weight` or `lm_head.weight`.
  - The two 412 MB embedding matrices exceed the 256 MiB shard budget, so the greedy packer must allow a single oversized tensor to sit alone in its shard.
- Anything in the task text or documentation that was unclear:
  - The shard file naming scheme and the key ordering used by the hidden reference are not specified; I used sorted key order and `model-{i:05d}-of-{n:05d}.safetensors`.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes
