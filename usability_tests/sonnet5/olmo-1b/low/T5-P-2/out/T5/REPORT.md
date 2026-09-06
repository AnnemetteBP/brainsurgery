## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 2 (1 failure, 1 success)
- Which executions failed, and why (one line each):
  - Execution 1: `FileNotFoundError` for `inputs/lora/adapter_config.json` — wrong number of `.parent` hops computing the sandbox root from `out/T5/solution.py`.
- Pitfalls or surprises you hit (one line each):
  - Path-depth arithmetic for locating `inputs/` relative to `out/T5/solution.py` was easy to get off by one.
  - Had to be careful that `B @ A` (no transpose) is correct here because `fan_in_fan_out = false` and both adapter factors and base weight already share the `[out, in]` `nn.Linear` layout.
  - Greedy shard-packing needed an explicit special case for tensors larger than the 512 MiB budget (`embed_tokens.weight`, `lm_head.weight`) so they get their own shard instead of blowing the budget or being skipped.
- Anything in the task text or documentation that was unclear: No — the layout convention and scale formula were stated explicitly, which avoided the usual ambiguity in these tasks.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~5 minutes
