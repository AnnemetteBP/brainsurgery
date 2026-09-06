## Participant self-report

- Final artifact path: `out/T1/plan.yaml` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1 failed: `assert.count` on `h\.(9|10|11)\..*` with `is: 0` raises
    `count.of matched zero tensors` instead of succeeding — `count` treats a
    zero-match reference as an error condition, not a valid count of 0.
- Pitfalls or surprises you hit (one line each):
  - `assert: count: { is: 0 }` cannot be used to prove absence, since the
    underlying reference resolver rejects zero matches before `count` gets to
    compare; used `assert: { not: { exists: ... } } }` instead.
  - Renumbering collisions: moving blocks in old-index ascending order
    (3->2, 4->3, 6->4, 7->5, 9->6, 10->7, 11->8) guarantees each destination
    index is already vacated (by the earlier `delete`s of blocks 2/5/8 or by
    the immediately preceding move) before the move that targets it runs, so
    `move`'s destination-must-not-exist rule never fires.
- Anything in the task text or documentation that was unclear:
  - None; the README's `count`/`exists`/`not` assert semantics and the
    structured `*rest` splice token in `move` covered everything needed.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes.
