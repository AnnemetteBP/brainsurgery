# T1 self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/plan.yaml` (output checkpoint: `out/T1/model.safetensors`, 86 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: `move` refuses existing destinations, so the moves are ordered by ascending old index (3->2, 4->3, ..., 15->11) so each destination is vacant when its block arrives.
  - Regex dots must be escaped and the index anchored with a trailing `\.` (e.g. `model\.layers\.1\.`), otherwise `layers.1` would also match `layers.10`..`layers.15`.
  - The output path ends in `.safetensors`, so a single file was written instead of a sharded directory (no `shard` setting needed).
- Anything in the task text or documentation that was unclear:
  - The required check "no tensor of blocks 12, 13, 14, 15 remains" refers to post-renumbering indices; I asserted it after the moves, and additionally asserted the removed blocks 2, 6, 10, 14 are gone right after the delete.
  - The README does not state whether a `move` with a regex source processes matches atomically; ordering the moves so destinations are always free sidesteps the question.
- Tools used (condition F): n/a (condition B, plan only). A read-only safetensors comparison was run afterwards to double-check the key set and bit-exact values; it is not part of the solution.
- Approximate time spent, if you can tell: about 5 minutes (reading the doc pack, writing the plan, one run of ~10 s).
