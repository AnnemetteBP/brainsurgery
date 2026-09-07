## Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `assert: { count: { of: 'h\.(9|10|11)\..*', is: 0 } }` raised `count.of matched
     zero tensors` because `count` resolves the reference before comparing and
     errors out on zero matches; switched to `assert: { not: { exists: ... } }`
     to express "no tensor of blocks 9/10/11 remains".
- Pitfalls or surprises you hit (one line each):
  - `move` requires the destination to not already exist, so the block
    renumbering has to be ordered so lower indices are freed (by delete or by
    an earlier move) before a higher block moves into them; doing the moves in
    increasing old-index order (3,4,6,7,9,10,11) makes every destination slot
    already vacant when its move runs.
  - `assert: count: is: 0` cannot express "matches nothing" — `count`
    resolution itself fails on zero matches; `not: { exists: ... }` is the
    right idiom for an absence check.
- Anything in the task text or documentation that was unclear: none; the
  README's structured-expression/regex-capture rewrite rule for `to`
  (same rule as `equal`'s `right`) made the per-block rename straightforward
  once combined with regex captures (`h\.3\.(.*)` -> `h.2.\1`).
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: a few minutes
