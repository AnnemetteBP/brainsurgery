# Participant self-report: T1 (Pythia-1B depth pruning), condition B

- Final artifact path: `out/T1/plan.yaml` (output checkpoint: `out/T1/model.safetensors`, 184 tensors; executed-plan summary in `out/T1/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Regex tensor references are full-match, so the layer-index dot must be escaped (`layers\.1\.` would otherwise not distinguish 1 from 10..15); I used an explicit alternation `(2|6|10|14)` for the delete.
  - Renumbering collision hazard: moves were ordered ascending by old index (3->2, 4->3, ..., 15->11) so each destination index was already vacated by the delete or by the previous move; `move` refusing an existing destination acts as a built-in collision guard.
  - Zero-match checks were expressed as `not: { exists: ... }` rather than `count: { is: 0 }`, since the docs do not say how `count` treats an empty match set.
- Anything in the task text or documentation that was unclear:
  - The README does not state the regex rewrite syntax for `to` in `move`/`copy` directly; it is only shown for `assert: equal` (`\1`) and in the interfaces reference ("from regex captures reused in to"). Worked as expected.
  - Whether `assert: count` with `is: 0` succeeds or raises a no-match error is not documented.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 3 minutes (docs reading, one plan execution, one read-only verification of the output against the input).
