# T3 participant self-report

- Final artifact path: `out/T3/solution.py` (output: `out/T3/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Over-broad targeting is the obvious trap: an anchored regex on
    `model.layers.<i>.(self_attn.[qkvo]_proj|mlp.(gate|up|down)_proj).weight`
    keeps `model.embed_tokens.weight` and `lm_head.weight` in float32.
  - The 256 MiB budget is on tensor data only, so the two 412 MB float32
    embedding matrices must each get a shard of their own; the greedy packer
    needs the "a single oversized tensor is allowed alone" carve-out.
  - Shard assignment depends on key order, so I used safetensors' own
    lexicographic order (which is how the input shards are laid out); it packs
    exactly two layers (128 MiB each after the bf16 cast) per 256 MiB shard.
  - `save_file` rejects shared/non-contiguous storage, so each shard's tensors
    are cloned before writing (the bf16 casts already copy, the fp32 ones did not).
- Anything in the task text or documentation that was unclear:
  - The shard file naming convention and the required key order within shards
    are not specified; I followed the HF `model-{i:05d}-of-{n:05d}.safetensors`
    convention and lexicographic key order.
  - "drop non-parameter buffers" in the objective is contradicted by the input
    description and requirement 3 (this checkpoint has none) — I deleted nothing.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~5 minutes.
