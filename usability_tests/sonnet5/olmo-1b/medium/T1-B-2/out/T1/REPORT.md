## Participant self-report

- Final artifact path: `out/T1/plan.yaml` (executed with `brainsurgery out/T1/plan.yaml`, output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `assert: { count: { of: 'model\.layers\.(12|13|14|15)\..*', is: 0 } }` raised `count.of matched zero tensors` instead of comparing to 0 — `count` requires at least one match, so it can't express "zero matches"; a plain crash, not a failed assertion.
- Pitfalls or surprises you hit (one line each):
  - `count` errors out on a zero-match reference rather than treating the match count as 0, so "assert nothing of X remains" has to be written as `assert: { not: { exists: ... } }` instead.
  - Renumbering old→new is not a uniform shift (0,1 stay; 3,4,5→2,3,4; 7,8,9→5,6,7; ...), so it can't be done with one regex substitution; used one `move` per surviving old index instead.
  - To avoid the move-order collision hazard, moves were issued in ascending old-index order — each source frees its old slot name before a later move needs that number as a destination, so `move`'s "destination must not exist" rule never trips.
- Anything in the task text or documentation that was unclear: none; the doc pack's `help: assert` and `help: move` sections were sufficient once the `count`-on-zero-matches behavior was discovered by trial.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: a few minutes of plan writing plus two `brainsurgery` invocations; verified bit-exactness against `inputs/base` afterwards with an ad hoc script.
