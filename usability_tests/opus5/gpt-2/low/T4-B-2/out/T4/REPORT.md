# T4 self-report (condition B, BrainSurgery plan)

- Final artifact path: `out/T4/plan.yaml` -> `out/T4/model.safetensors`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `no_match` — `assert: count: { of: 'base::tv\d\..+', is: 0 }` (my "scratch tensors are gone" check) failed with "count.of matched zero tensors"; `count` resolves its reference first and errors on zero matches, so it cannot assert absence. Removed that one assert; everything before it (all verification asserts and the whole merge) had already passed.
- Pitfalls or surprises you hit (one line each):
  - `count: is: 0` is unusable as an absence check — zero matches is a hard resolver error, not a count of 0.
  - With three inputs, the output alias is inferred from what the transforms *write*, so both task vectors had to be staged as scratch tensors on the `base` alias (`copy: from: 'ft1::(...)', to: 'base::tv1.\1'`) rather than scaled in place on `ft1`/`ft2`.
  - Ordering hazard handled by staging both task vectors and subtracting the base from both *before* any `add_` touches the base MLP tensors.
  - `equal` with a negative lookahead (`left: 'base::(?!h\.\d+\.mlp\.).+', right: 'ft1::\g<0>'`) covers the shared-tensor precondition in one line; I ran it in both directions so a renamed/extra tensor in a fine-tune also fires.
  - Name-level checking of the 48 MLP tensors comes for free from `copy`/`subtract_`/`add_`: a missing or renamed destination aborts the run.
- Anything in the task text or documentation that was unclear:
  - The README does not say that `count` (and target resolution generally) treats zero matches as an error; that cost the one retry.
  - It is not spelled out whether `to` in `add_`/`subtract_` is a regex rewrite of the `from` matches the way `copy`'s `to` is; it is, but I had to infer it.
- Tools used (condition F): n/a (condition B; only `brainsurgery` and the doc pack).
- Approximate time spent, if you can tell: ~10 minutes.
