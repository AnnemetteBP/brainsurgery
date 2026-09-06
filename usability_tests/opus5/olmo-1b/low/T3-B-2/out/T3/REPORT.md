# T3 self-report (condition B)

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Over-broad targeting is the obvious trap: `.*weight` would also hit
    `model.embed_tokens.weight` and `lm_head.weight`, so the cast pattern spells out
    `model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight`.
  - Shard budget units are binary in BrainSurgery, so `256MB` is exactly 268,435,456 bytes.
  - The two 412 MB float32 tensors exceed the budget and are each written alone in their own
    shard, which the documented packing rule does automatically.
  - I added a negative-lookahead assert (count 2 + dtype float32) so the run fails if the cast
    ever leaked onto a non-projection tensor, not just if the intended 112 were cast.
- Anything in the task text or documentation that was unclear:
  - The task says "drop non-parameter buffers" in the objective but the input has none and
    requirement 3 says not to delete anything; I deleted nothing.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~5 minutes.
