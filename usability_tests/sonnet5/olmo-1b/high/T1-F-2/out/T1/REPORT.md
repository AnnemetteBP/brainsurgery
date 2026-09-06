# T1 report (condition F)

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), producing `out/T1/model.safetensors`.
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The renumbering map is not a simple arithmetic shift once blocks are dropped mid-sequence (e.g. old 11 -> new 8, old 12 -> new 9), so I built an explicit old-to-new dict from a single ordered pass over `range(16)` excluding the dropped set, rather than computing an offset per block, to avoid a collision if I moved blocks out of order.
  - The two shards had to be loaded via the index's `weight_map` rather than assumed to split cleanly by block, since block tensors can straddle shard boundaries.
- Anything in the task text or documentation that was unclear: none; the explicit old->new index list in the task made the required mapping unambiguous.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, as a plain script — no `mergekit` or `torch-state-bridge` needed since this is a straightforward filter-and-rename over a flat state dict with no shape changes, task-arithmetic, or fuzzy key matching involved; a plain script also made it easy to write the exact bit-equality and key-set checks the task requires directly against the in-memory tensors before anything is written to disk.
- Approximate time spent, if you can tell: ~10 minutes including verification.
