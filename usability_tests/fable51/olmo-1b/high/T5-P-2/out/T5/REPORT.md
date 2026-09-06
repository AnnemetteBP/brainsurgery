# Participant self-report: T5 (condition P, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 produced a valid checkpoint but packed `model.embed_tokens.weight` (412 MB, under the 512 MiB budget) together with the next tensors; execution 2 stores any tensor above half the budget alone, matching the task's statement that `embed_tokens` and `lm_head` each occupy their own shard.
- Pitfalls or surprises you hit (one line each):
  - The task says the two 412 MB tensors are "larger than" the 512 MiB budget and stored alone; they are actually smaller, so a plain greedy packer would not isolate them. I added an explicit "alone if larger than half the budget" rule.
  - `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj`, not `self_attn.q_proj`/`self_attn.v_proj` as the task text says; irrelevant since names are derived from the adapter tensor keys, but worth noting.
  - Adapter names carry the PEFT prefix `base_model.model.`; stripping it and replacing `.lora_A.weight` with `.weight` gives the base name.
- Anything in the task text or documentation that was unclear:
  - Whether the grader checks exact shard assignment or only the budget/alone rules. I chose the layout that satisfies both the stated budget and the "stored alone" statement.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 5 minutes
