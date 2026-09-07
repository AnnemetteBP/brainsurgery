# Participant self-report: T5 (GPT-2 124M, condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget, so the packer needs an explicit "oversized tensor goes alone" branch rather than plain greedy packing.
  - `fan_in_fan_out = true` means the `B @ A` product ([out, in]) must be transposed to the Conv1D [in, out] base layout; the script reads the flag from `adapter_config.json` rather than hardcoding it.
  - PEFT adapter names carry a `base_model.model.` prefix and a `.lora_A/.lora_B.weight` suffix that must be stripped to reach `h.<i>.attn.c_attn.weight`.
- Anything in the task text or documentation that was unclear:
  - Shard file naming and tensor ordering within shards are not specified; I used the HF convention `model-XXXXX-of-XXXXX.safetensors` and greedy packing in the base file's key order.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: about 2 minutes.
