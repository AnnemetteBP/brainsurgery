## Participant self-report

- Final artifact path: out/T1/plan.yaml
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each): renumbering must go in increasing source order (3,4,5,7,8,9,11,12,13,15) so each destination slot is already free (either deleted or already vacated by an earlier move) before it is written; regex targets need explicit `\d+`/anchored digit groups so e.g. `layers.1.` does not also match `layers.11.` or `layers.13.`.
- Anything in the task text or documentation that was unclear: none; the docpack's move/delete/assert help entries and the regex full-match semantics were sufficient.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes.
