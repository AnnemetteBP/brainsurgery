# Participant self-report

- Final artifact path: out/T3/plan.yaml
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - Ordering the transforms as delete-buffers -> upcast-everything-to-float32 ->
    downcast-projections-to-bfloat16 avoided needing an exclusion regex for the
    "everything else" float32 pass.
  - `assert.dtype`/`assert.count` take a single tensor-ref that can match many
    tensors, so `count`+`dtype` on the same projection-matrix regex together
    encode "exactly 64 tensors, all bfloat16" without a dedicated
    count-tensors-of-a-given-dtype assertion.
- Anything in the task text or documentation that was unclear: none; the
  README's note on sharding (single oversized tensor gets its own shard) matched
  the observed output exactly.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes, single pass, no retries.
