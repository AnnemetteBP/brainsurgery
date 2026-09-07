# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The projection-matrix regex has to be anchored and dot-escaped
    (`^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$`)
    so it only ever hits the 112 intended matrices, never `embed_tokens.weight` or
    `lm_head.weight`.
  - Bin-packing tensors by insertion order (raw `weight_map` dict order) would have
    produced ugly/uneven shards since layer 10 sorts before layer 2 as a string; sorted
    the tensor names numerically by layer index before packing so shards read as
    contiguous layer ranges.
  - With bf16, exactly two full transformer layers (4 attn + 3 mlp matrices each) sum to
    precisely 256 MiB, so the shard budget lines up neatly with 2-layer boundaries here;
    that's incidental to this checkpoint's shapes, not something the script assumes.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~10 minutes.
