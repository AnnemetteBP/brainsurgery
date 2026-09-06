# T3 self-report (Condition B: BrainSurgery plan)

- Final artifact path: `out/T3/` (10 shards + `model.safetensors.index.json`);
  plan at `out/T3/plan.yaml`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed
  all asserts and wrote the checkpoint.
- Pitfalls or surprises you hit (one line each):
  - The obvious over-broad pattern `.*weight` would also catch
    `model.embed_tokens.weight` and `lm_head.weight`; I anchored the regex on
    `model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight`
    and verified it matches exactly 112 names before and after the cast.
  - "Exactly 112 tensors are bfloat16" is not expressible as one assert, so I
    combined `count`/`dtype` on the projection pattern with a negative-lookahead
    `dtype: float32` over everything else plus `count: {of: '.*', is: 114}`.
  - Shard units are binary in this tool, so `shard: 256MB` is exactly the
    268,435,456-byte budget the task asks for; the two 412 MB float32 tensors
    were automatically placed alone in their own shards.
  - This checkpoint has no norms, biases or buffers, so no `delete` was needed
    and nothing was renamed.
- Anything in the task text or documentation that was unclear:
  - The task objective mentions dropping buffers and upcasting norms/biases, but
    the Input section states this checkpoint has none of those; the two
    statements read as contradictory until the Required-result list resolves it.
  - The README documents binary shard units clearly, but it is worth noting the
    budget counts tensor data only, not headers (my shards are 268,437,048 bytes
    on disk against 268,435,456 bytes of tensor data).
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~5 minutes, most of it reading
  `docpack/README.md` and the `help` output for `cast_` and the assert operators.
