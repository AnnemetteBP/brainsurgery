# T5 participant self-report (condition P, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 succeeded but greedy packing in base index order placed `model.layers.0.mlp.down_proj.weight` in the same shard as `model.embed_tokens.weight` (479 MB, within budget); the task text says the two 412 MB tensors are stored alone, so I added a rule that tensors of at least 256 MiB get their own shard and re-ran (execution 2).
- Pitfalls or surprises you hit (one line each):
  - The sharding rule as written ("a single tensor larger than 512 MiB is stored alone") does not by itself force the 412 MB embedding and lm_head into their own shards; the parenthetical example does. I followed the example.
  - `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj` (no `self_attn.` prefix) while TASK.md gives the prefixed form; I paired A/B by name pattern instead of using `target_modules`, so this did not matter.
  - PEFT prefix `base_model.model.` must be stripped and `.lora_A.weight` replaced by `.weight` to reach the base name.
- Anything in the task text or documentation that was unclear:
  - The "stored alone" sharding rule vs. the 412 MB sizes (see above). Whether the reference packs greedily in base index order is not stated; I used base index order.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes
