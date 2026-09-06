## Participant self-report

- Final artifact path: `out/T1/plan.yaml` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `failed_assertion` / `crash` at resolution time: `assert: { count: { of: '...(12|13|14|15)...', is: 0 } }` raised
     `count.of matched zero tensors: ...` instead of evaluating the comparison, because the tensor-ref
     resolver raises unconditionally on zero matches before `count`'s own `is` check runs — so `count`
     can never assert "matches exactly zero".
- Pitfalls or surprises you hit (one line each):
  - `assert: count` cannot express "this pattern matches nothing" because the underlying reference
    resolver treats zero matches as an error regardless of the expected count; had to switch that
    check to `assert: { not: { exists: <pattern> } } }`, which works because `not` catches the
    `TransformError` the resolver raises and inverts it.
  - Renumbering with `move` has a collision hazard if done out of old-index order: doing the moves in
    increasing old-index order (3→2, 4→3, 5→4, 7→5, 8→6, 9→7, 11→8, 12→9, 13→10, 15→11) after deleting
    blocks 2/6/10/14 first guarantees every destination slot is already empty when its move runs, so no
    explicit intermediate/tmp renames were needed.
  - Structured path tokens (`["gpt_neox","layers","3","*rest"]`) made `delete`/`move` targets exact and
    avoided any risk of a loose regex accidentally matching, e.g., layer `13` when targeting layer `1`.
- Anything in the task text or documentation that was unclear:
  - The doc pack doesn't call out that `assert.count`/`assert.exists` (and any tensor-ref resolution in
    general) fail hard on zero matches rather than returning an empty set — this makes "assert absence"
    only expressible via `not: { exists: ... } }`, which isn't stated anywhere in `help.txt` or the
    README's assert examples.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes
