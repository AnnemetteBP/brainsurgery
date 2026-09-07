## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the plan succeeded on the first run.
- Pitfalls or surprises you hit (one line each):
  - `cast_` takes a `to` key (not `dtype`) for the target dtype.
  - `assert.exists` payload is a bare tensor-ref, not a mapping with `of` (had to wrap it under `not` as `not: { exists: <ref> }`).
  - Ordered the transforms as delete buffers -> upcast everything to float32 -> downcast the 64 projection matrices to bfloat16, so the final cast overrides the blanket float32 upcast only where intended.
- Anything in the task text or documentation that was unclear: none; the doc pack's transform/assert help text had the exact required keys.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes, single pass.
