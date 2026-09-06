# Participant self-report

- Final artifact path: `out/T1/plan.yaml` (run with `brainsurgery out/T1/plan.yaml`), producing `out/T1/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering isn't a uniform arithmetic shift (2 blocks are skipped every
    4 layers at different offsets), so a single regex substitution can't
    express it; each shifted block needs its own `move` with a capture on the
    per-tensor suffix (`model\.layers\.<old>\.(.*)` -> `model.layers.<new>.\1`).
  - `move` destinations must not already exist, so the 10 renumbering moves
    have to run in ascending old-index order: each move's target slot is
    either one of the 4 deleted blocks or a slot vacated by an earlier move
    in the same list, so no move ever collides with survivor data.
  - Deleting the 4 pruned blocks before doing any moves avoids relying on
    delete/move ordering relative to each other for the two slots (2 and 10)
    that both get freed by deletion and reused as move targets.
- Anything in the task text or documentation that was unclear: none; the
  README's tensor-reference and `move`/`delete` help text sections were
  sufficient to build the plan without any trial-and-error executions.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes (documentation review
  + plan authoring + verification), single execution.
