## Participant self-report

- Final artifact path: `out/T4/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run
- Pitfalls or surprises you hit (one line each):
  - `subtract`/`add`/`copy` all require existing (or non-existing, respectively) destinations, so the task vector had to be built with `copy` (create) then `subtract_` (in-place `to -= from`) rather than a single `subtract`.
  - Keeping the arithmetic anchored to the unmodified base required computing both `delta1 = ft1 - base` and `delta2 = ft2 - base` as scratch tensors *before* mutating `base` in place with `add_`, otherwise the second task vector would have been taken against an already-merged base.
  - `output` alias inference needs every write to land on a single alias, so all copy/subtract_/scale_/add_/delete transforms target `base::...` explicitly, keeping `ft1`/`ft2` untouched.
- Anything in the task text or documentation that was unclear: none; the README's `assert.equal` example with a negative-lookahead pattern (`left: 'a::(?!h\.\d+\.mlp\.).+', right: 'b::\g<0>'`) mapped directly onto the "verify everything outside the 48 MLP tensors is identical" requirement.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes
