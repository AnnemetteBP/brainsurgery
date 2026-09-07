# T5 participant self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T5/solution.py` (output checkpoint: `out/T5/model-0000{1..5}-of-00005.safetensors` + `out/T5/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `adapter_config.json` lists `target_modules = ["c_attn"]` (not `attn.c_attn` as the task text says); the script derives the target from the adapter tensor name instead, so this did not matter.
  - The base checkpoint has no `transformer.` prefix, while PEFT names carry `base_model.model.`; stripping that prefix and appending `.weight` maps directly.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget and must sit alone in its shard; the greedy packer handles it as a single-tensor shard.
- Anything in the task text or documentation that was unclear: the exact shard file naming and packing order are not specified; I used HF-style `model-XXXXX-of-XXXXX.safetensors` names and greedy packing in base key order.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load base and adapter tensors, `save_file` for each shard (no model instantiation needed).
  - `torch` 2.14.0: float32 matmul `B @ A`, scale by alpha/r = 2, transpose for `fan_in_fan_out = true` (Conv1D `[in, out]` base), add.
  - I did not use `peft.merge_and_unload` or `transformers.save_pretrained`: they would instantiate the model, rename keys (`transformer.` prefix, tied `lm_head`), and give less control over the shard budget rule and required checks; a direct script is shorter and transparent.
- Approximate time spent, if you can tell: about 2 minutes.

## Checks enforced in `solution.py` (before writing)
- exactly 12 adapter pairs merged (and adapter tensor count = 2 x pairs);
- no output name contains `lora_`;
- `h.0.attn.c_attn.weight` is `[768, 2304]` float32;
- output has exactly 160 tensors (same count as base);
- each shard holds at most 100 MiB of tensor data unless it is a single tensor.

Post-run verification (separate check, not part of the artifact): all 148 non-adapted tensors bit-exact with base, the 12 merged weights equal `W + 2 * (B @ A).T` exactly, shard data sizes 101.8 / 104.7 / 100.0 / 87.1 MB plus `wte.weight` alone at 154.4 MB.
