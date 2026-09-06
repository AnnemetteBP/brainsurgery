# Participant self-report

- Final artifact path: `out/T4/solution.py` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base checkpoint is a sharded safetensors directory (index json + two
    shard files) while the fine-tunes are single files, so the loader needs
    to handle both layouts.
  - Had to be careful to compute each fine-tune's task vector against the
    unmodified `base` tensor rather than against a partially-updated
    accumulator, since `base + lambda*(ft1-base) + lambda*(ft2-base)` is not
    the same as sequentially updating `base` in place.
- Anything in the task text or documentation that was unclear: no — the spec
  (shared-tensor precondition, merge formula, required checks) was
  unambiguous.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes; single-pass script,
  no debugging needed.
