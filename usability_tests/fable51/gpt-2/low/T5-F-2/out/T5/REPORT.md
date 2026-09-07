# T5 participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - PEFT could not consume this adapter as-is (name prefix mismatch, only a warning), which would make a naive `merge_and_unload` route produce an unmerged output.
  - `fan_in_fan_out = true` was handled by transposing `B @ A` before adding, and a greedy shard packer with an "oversized tensor alone" rule met the 100 MiB budget (5 shards, `wte.weight` alone in the last).
- Anything in the task text or documentation that was unclear: nothing.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load base and adapter, save shards (direct control over shard byte budget, which `save_pretrained` does not expose exactly).
  - `torch` 2.14.0: float32 matmul, transpose, add.
  - `numpy` 2.5.2: post-hoc cross-check in float64 only (merged weights agree with the script to ~1e-8 relative error); not part of the solution.
  - `peft` 0.20.0 was tried for a cross-check but silently skipped the adapter (its keys lack the `transformer.` prefix PEFT expects for GPT-2), so `merge_and_unload` returned the unmodified base; not used.
- Approximate time spent, if you can tell: about 2 minutes.
