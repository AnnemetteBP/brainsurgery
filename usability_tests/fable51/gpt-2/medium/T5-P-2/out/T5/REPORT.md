# Participant self-report: T5 (GPT-2 124M), condition P

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Adapter names carry the PEFT prefix `base_model.model.`; stripped it and appended `.weight` to reach the base name.
  - `fan_in_fan_out = true` means `B @ A` is `[out, in]` and must be transposed to the Conv1D `[in, out]` layout before adding; checked the delta shape against the weight shape before adding.
  - Greedy sharding in file order packs 4 shards of about 100 MiB; `wte.weight` (154 MB) exceeds the budget so it lands alone in the last shard.
  - Saved merged tensors with `.contiguous()` since the transposed delta is a view.
- Anything in the task text or documentation that was unclear: the shard file naming scheme is not specified; used the HF convention `model-0000i-of-0000n.safetensors`. Sharding order (file order vs. sorted names) is also unspecified; used file order.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes
