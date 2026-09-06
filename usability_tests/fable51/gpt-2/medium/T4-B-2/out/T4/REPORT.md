# T4 (GPT-2 124M), condition B: participant self-report

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - With three inputs, brainsurgery infers the output alias from the alias the transforms write to, so every edit (copies, in-place ops, deletes) had to land on `base::`; the task vectors were computed into temporary `base::tv1.*` / `base::tv2.*` names instead of editing `ft1`/`ft2`.
  - `subtract`/`add` require an existing destination, so the task vectors were built as `copy` (ft -> tvN) followed by in-place `subtract_` of the untouched base, then `scale_`, then `add_` into base, then `delete` of the temporaries.
  - Shared-tensor verification used `assert: equal` with a negative-lookahead regex (`(?!h\.\d+\.mlp\.).+`) and `\g<0>` rewrite to compare base vs ft1 and base vs ft2 name-for-name; exact name-set equality was pinned with `count` checks (160 total, 48 MLP names in each checkpoint).
  - Ordering: both task vectors were computed before any `add_` touched the base, avoiding the "second vector measured against a modified base" hazard.
- Anything in the task text or documentation that was unclear:
  - The help text for the in-place transforms (`add_`, `subtract_`) only shows literal-name examples; that `to` is a capture-group rewrite of `from` (like `copy`) is stated in the interfaces reference, not in the per-transform help.
  - The plan-level check "exactly 48 tensors were merged" cannot be asserted directly on a transform's match count; it was approximated by asserting 48 task-vector tensors per fine-tune (and 256 tensors total) right before the merge, and 160 tensors after cleanup.
- Tools used (condition F): n/a (condition B). After the run, a short read-only Python check confirmed 112 bit-exact tensors and zero relative error on the 48 merged tensors; it is not part of the solution.
- Approximate time spent, if you can tell: about 5 minutes (one plan run of ~11 s).
