## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - Needed to write the projection-matrix regex explicitly
    (`model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight`)
    rather than a broad `.*weight` pattern, since that would also match
    `model.embed_tokens.weight` and `lm_head.weight`.
  - Used `cast_` (in-place) instead of `cast` since the task requires tensor
    names to stay unchanged and `cast` always creates a new destination
    tensor.
  - `output.shard: 256MB` was sufficient to get the required per-shard budget
    (256 MiB = 268,435,456 bytes); the two oversized float32 tensors
    (embed_tokens, lm_head) each landed alone in their own shard automatically,
    as documented.
- Anything in the task text or documentation that was unclear: none; the
  README's notes on shard sizing (binary units, largest-tensor-alone rule)
  matched the observed output exactly.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes.
