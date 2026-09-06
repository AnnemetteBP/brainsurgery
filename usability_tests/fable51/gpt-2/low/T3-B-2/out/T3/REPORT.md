# Participant self-report: T3 (condition B, GPT-2 124M)

- Final artifact path: `out/T3/plan.yaml` (output shards + `model.safetensors.index.json` in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all asserts and wrote 148 tensors across 4 shards.
- Pitfalls or surprises you hit (one line each):
  - No dtype-based tensor selector exists, so "exactly 48 tensors are bfloat16" was expressed as: count of the projection pattern is 48, all of those are bfloat16, and the complement pattern (negative lookahead) is 100 tensors, all float32.
  - Regex references are full-match, so the projection pattern had to be anchored via the `\d+` layer index and explicit `weight` suffix to avoid hitting `c_attn.bias` etc.
  - `shard: 64MB` uses binary units (64 MiB), matching the task's 67,108,864 bytes; `wte.weight` correctly landed alone in its own shard.
- Anything in the task text or documentation that was unclear: the `help.txt` example `cast_: { target: '.*weight' }` is exactly the over-broad pattern the task warns about; otherwise clear.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes.
