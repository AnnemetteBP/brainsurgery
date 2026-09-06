## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1 failed: `assert: { count: { of: 'base::merge_tmp\..*', is: 0 } }` raised
    `count.of matched zero tensors` instead of evaluating to a count of 0 — `count`
    errors on zero matches rather than treating it as a valid count; switched that
    check to `assert: { not: { exists: ... } } }`. No output was written by this
    failed attempt (the failure happened before the save stage).
- Pitfalls or surprises you hit (one line each):
  - `count`'s `of` reference raises an error on zero matches instead of succeeding
    with count 0, so "assert nothing left" must be written as `not: { exists: ... }`.
  - Had to confirm from `docpack/interfaces-reference.md` (not the README) that
    ternary transforms (`matmul`, etc.) resolve `from_a` as the matched/capturing
    reference while `from_b`/`to` are rewrites reusing its capture groups, rather
    than being independently pattern-matched and paired positionally.
- Anything in the task text or documentation that was unclear:
  - None; the doc pack's ternary-transform capture-rewrite note and the sharding
    section were sufficient to get the plan right on the first structural attempt.
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: ~15 minutes
