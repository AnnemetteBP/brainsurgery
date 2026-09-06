# Participant self-report: T4 (condition B)

- Final artifact path: `out/T4/plan.yaml` (output `out/T4/model.safetensors`, 114 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: my own `assert: writes` check on the non-MLP tensors fired (`failed_assertion`); the access counter counts the initial load as one write, so `is: 0` was wrong (fixed to `is: 1` for untouched tensors, `is: 3` for merged ones).
- Pitfalls or surprises you hit (one line each):
  - `writes` access counts include the initial load, not only plan edits.
  - With several inputs the output alias is inferred from write destinations, so task vectors had to be built as temporaries under the `base` alias (`base::tv1.<name>`) and deleted afterward.
  - No non-in-place subtract into a new tensor exists, so `copy` + `subtract_` was used to form each task vector against the unmodified base.
- Anything in the task text or documentation that was unclear:
  - The `writes`/`reads` semantics (what counts as a write) are not documented in the README.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes
