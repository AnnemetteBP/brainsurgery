# Participant self-report: T2 (Pythia-1B head pruning, condition B)

- Final artifact path: `out/T2/plan.yaml` (output checkpoint: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `failed_assertion`/`no_match` on my own extra sanity check `assert: { count: { of: '.*\.pruned', is: 0 } }`; `count` treats a zero-match reference as an error (`count.of matched zero tensors`) rather than counting 0. All pruning transforms and all four required checks had already passed; no output was written. Replaced with `assert: { not: { exists: '.*\.pruned' } }`.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source ref to resolve to exactly one tensor, so the plan cannot use one pattern over all 16 layers; it needs 16 x 3 explicit concat groups (generated with a shell loop, 150 transforms total).
  - `concat` only creates new tensors, so keeping the original name needs a three-step concat-to-temp, delete original, move-back sequence per tensor.
  - `count` with `is: 0` is not usable as "no tensor matches"; use `not: exists` instead.
- Anything in the task text or documentation that was unclear:
  - The help text for `concat`/`split` shows `to: ` with an empty value in the examples, which is unhelpful; the README's `[:, :4]` example was what confirmed the slice syntax.
  - Nothing in the docs says that a zero-match reference in `assert: count` raises instead of evaluating to 0.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
