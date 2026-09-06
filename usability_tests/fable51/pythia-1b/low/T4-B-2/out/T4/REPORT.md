# T4 (Pythia-1B), condition B: participant self-report

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - With several inputs the output alias is inferred from write targets, so all scratch tensors (`f32.*`, `d1.*`, `d2.*`) had to be created inside the `base` alias with `base::` destinations.
  - Merged values must end up under the original names, so the merged float16 tensors are swapped in with `delete` of the originals followed by `move`, since `cast`/`move` refuse existing destinations.
  - Name-set equality is expressed indirectly: equal `count` per alias plus `equal` (every left match must map to an existing right tensor) on the non-MLP subset; MLP name/shape/dtype agreement is only enforced implicitly by the arithmetic transforms failing on mismatch.
- Anything in the task text or documentation that was unclear:
  - The `add`/`subtract` help does not state explicitly that `from_b` and `to` are rewrites of each `from_a` match with capture groups; it worked as in `copy`/`equal`, but I had to guess.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
