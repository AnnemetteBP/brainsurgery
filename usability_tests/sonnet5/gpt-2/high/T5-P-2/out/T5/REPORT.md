# Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base weight uses Conv1D `[in, out]` layout while `B @ A` produces `[out, in]`, so the transpose is required before adding — got this right from `fan_in_fan_out=true` in the adapter config on the first try.
  - `target_modules` in `adapter_config.json` is `["c_attn"]`, not `["attn.c_attn"]`; the adapter tensor names still carry the full `attn.c_attn` path, so the regex over adapter keys handled it without needing to special-case the config value.
  - The 100 MiB shard budget is on tensor payload bytes only, so I verified shard sizes by summing `numel * element_size` per tensor rather than trusting on-disk file size (safetensors headers add a few KB per shard).
- Anything in the task text or documentation that was unclear: none; the shapes, scale formula, and shard-budget exception for the oversized `wte.weight` tensor were stated explicitly enough to implement directly.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: one pass of writing + one execution + independent verification, on the order of a few minutes of wall-clock work.
