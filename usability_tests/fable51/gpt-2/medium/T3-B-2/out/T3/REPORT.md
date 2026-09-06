# Participant self-report: T3 (GPT-2 124M, condition B)

- Final artifact path: `out/T3/plan.yaml` (output shards + `model.safetensors.index.json` in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `assert.count` cannot filter by dtype, so "exactly 48 bfloat16" is expressed as: the 48-matrix pattern matches 48 tensors, all of them are bfloat16, and every tensor outside that pattern (negative-lookahead regex) is float32.
  - The 48-matrix regex must be anchored on the exact submodule names so `h.<i>.attn.bias` (the mask buffer) and the projection biases are not touched.
  - The oversized `wte.weight` (154 MB) lands alone in the last shard, as the docs describe; the other three shards are under the 64 MiB budget.
- Anything in the task text or documentation that was unclear: whether `assert.dtype` checks all matches of a multi-match pattern; it did (no error on the 48-match and the negated pattern). The README's `shard` units are binary (`64MB` = 64 MiB), which matched the task's 67,108,864-byte budget.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes
