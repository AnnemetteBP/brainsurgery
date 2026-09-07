# Participant self-report (T2, condition B, GPT-2 124M)

- Final artifact path: `out/T2/plan.yaml` (output checkpoint `out/T2/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 1 (plus one separate read-only verification plan against the written output, containing only asserts and a dump)
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `concat` destinations must not already exist, so each pruned tensor is built under a `tmp.` name, the original is `delete`d, then `move`d back to its original name (3 transforms per tensor, 9 per layer, 108 total).
  - No loop construct in plans, so the per-layer transforms were generated explicitly for all 12 layers.
  - Output path given with a `.safetensors` suffix wrote a single unsharded file as required.
- Anything in the task text or documentation that was unclear:
  - Whether `assert.count` with `of: '.*'` counts the whole model was not spelled out; it did (160).
  - The docs do not say explicitly whether `move` may reuse a name freed by `delete` in the same plan; it works.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
