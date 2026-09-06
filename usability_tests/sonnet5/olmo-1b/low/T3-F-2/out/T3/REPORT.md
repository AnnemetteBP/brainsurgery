## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The two >256MiB tensors (`model.embed_tokens.weight`, `lm_head.weight`) need
    their own dedicated shard rather than being force-fit under the budget;
    handled with an explicit oversized-tensor branch in the bin-packing loop.
  - Non-cast tensors must remain bit-identical, not just "float32" — added an
    explicit `torch.equal` check against the loaded input tensors before writing.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 directly
  (`safe_open`/`save_file`), no higher-level toolkit. The task's precision
  targeting is a small, well-specified regex (per-layer projection names) plus
  an explicit whitelist of exceptions (embeddings, lm_head), so writing it as a
  plain script gave exact, auditable control over which tensors get cast and
  how shards are bin-packed to the byte budget — more direct than adapting
  `mergekit`/`transformers` sharded-save machinery to a non-default shard size
  and a manual dtype-per-tensor policy.
- Approximate time spent, if you can tell: ~10 minutes.
