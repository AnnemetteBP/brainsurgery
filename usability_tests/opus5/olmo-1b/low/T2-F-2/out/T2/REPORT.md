# T2 self-report

- Final artifact path: `out/T2/solution.py` -> `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Only real risk is the axis split: q/k/v are row blocks, o_proj is a column block; a regex on `self_attn` alone would have hit o_proj on the wrong axis, so the two cases are matched by separate anchored regexes.
  - `index_select` output must be made contiguous before `save_file`.
  - The input is sharded; shards are merged into one state dict and checked against the index weight map before slicing.
- Anything in the task text or documentation that was unclear: nothing; the keep-row ranges and required shapes were fully specified.
- Tools used (condition F):
  - `safetensors` 0.5.3 — shard load and single-file save; the task's output is one file with exact key/dtype preservation, which `save_file` does directly.
  - `torch` 2.14.0 — `index_select` with an explicit keep-index built from head geometry, so the kept block order is the spec's order by construction.
  - Not used: `transformers.prune_heads` (it prunes a live model and re-serializes via `save_pretrained`, which risks config/head-count rewriting, tied-weight handling and sharding decisions that would break the "114 tensors in one file, bit-exact, names unchanged" requirement); `mergekit` (layer-level, no intra-tensor head slicing).
- Approximate time spent, if you can tell: ~5 minutes.
