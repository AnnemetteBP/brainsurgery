# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Upcasting the two embedding tensors from float16 to float32 doubles their
    size to ~412 MB, well over the 256 MiB shard budget, so they still land
    alone in their own shard even though the task text describes them as
    "206 MB each" (that figure is the float16 input size, not the float32
    output size) — worth checking against the actual output dtype rather
    than the input size when reasoning about shard sizing.
  - Had to be careful to enumerate the buffer names explicitly
    (`attention.bias`, `attention.masked_bias`, `attention.rotary_emb.inv_freq`)
    rather than pattern-matching on `bias`, since `attention.bias` is a
    buffer but `attention.query_key_value.bias` and other projection
    biases are parameters that must be kept (and upcast to float32, not
    dropped).
- Anything in the task text or documentation that was unclear: the "206 MB
  each" figure for the embedding tensors is the input (float16) size; the
  actual float32 output size is about double that, which is worth stating
  explicitly to avoid confusion about the oversized-shard rule.
- Tools used (condition F): n/a (condition P — standalone PyTorch script).
- Approximate time spent, if you can tell: ~10 minutes.
