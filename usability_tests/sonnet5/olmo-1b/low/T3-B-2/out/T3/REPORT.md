## Participant self-report

- Final artifact path: `out/T3/` (10 safetensors shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - The only real hazard was making the target regex for the 112 projection
    matrices precise enough to exclude `model.embed_tokens.weight` and
    `lm_head.weight`; anchoring on the `model.layers.<i>...` prefix with the
    `(self_attn.(q|k|v|o)_proj|mlp.(gate|up|down)_proj).weight` suffix does
    that naturally since neither excluded tensor has a `layers` segment.
- Anything in the task text or documentation that was unclear: none; the
  README's shard-size section stating shard units are binary (`1MB` =
  1024x1024 bytes, budget counts tensor data only) made it straightforward
  to pick `shard: 256MB` to match the 268,435,456-byte requirement exactly.
- Tools used (condition F): n/a (condition B, plan-only).
- Approximate time spent, if you can tell: ~10 minutes.
