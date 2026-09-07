# Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `matmul`/`cast`/`add_` all require plain regex or structured references, so intermediate
    tensors (`delta`, `delta_f16`) had to be written under the same alias (`base`) as the final
    output, otherwise the output-alias inference would have failed with two aliases written to.
  - `add_` needs source and destination dtypes to match, so the float32 `B @ A * scale` delta had
    to be cast down to float16 before the in-place add into the (float16) base weight.
  - Intermediate tensors must be deleted before saving since they'd otherwise inflate the tensor
    count and appear in the output.
- Anything in the task text or documentation that was unclear: none; the README's ternary-transform
  capture-rewrite semantics (`from_a` matches, `from_b`/`to` are rewrites of its captures) were not
  spelled out in the `matmul` help text itself but were confirmed from the `interfaces-reference.md`
  note and the worked `copy`/`assert equal` examples in the doc pack.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes, single attempt.
