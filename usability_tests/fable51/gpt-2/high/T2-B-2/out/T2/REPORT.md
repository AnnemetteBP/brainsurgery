# Participant self-report: T2 (GPT-2 124M), condition B

- Final artifact path: `out/T2/plan.yaml` (output checkpoint `out/T2/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion on my own optional extra check `assert: { count: { of: 'pruned\..*', is: 0 } }`. The resolver raises "count.of matched zero tensors" before comparing the count, so a zero-match count assert can never pass. All edits and the four required checks had already succeeded; the failure happened before the write, so no partial output. Fixed by dropping the redundant line (the 160-tensor count already proves no leftover temporaries).
- Pitfalls or surprises you hit (one line each):
  - `concat` requires every `from` reference to resolve to exactly one tensor, so the per-layer slicing cannot be expressed with a single pattern; the plan has one concat per (layer, tensor), 36 in total, generated with a shell loop.
  - No in-place slicing/replace transform, so the flow is concat into temp names -> delete originals (anchored regex, avoiding `mlp.c_proj` and the `attn.bias` mask buffer) -> `move` back with regex captures.
  - `count` with `is: 0` errors on zero matches instead of passing; `not: exists` is the likely alternative but I did not test it.
  - `shape` help says "the tensor" (singular), so I did not rely on pattern targets for shape asserts and wrote one assert per layer instead.
- Anything in the task text or documentation that was unclear:
  - Whether pattern references that match several tensors are accepted by `shape`/`dtype` asserts (help text is singular; README is silent).
  - That zero-match references are a hard error for `count` is not documented.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
