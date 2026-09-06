# Participant self-report: T5 (Pythia-1B, condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `adapter_config.json` lists `target_modules = ["query_key_value"]`, not `attention.query_key_value` as in TASK.md; the script maps names by stripping the `base_model.model.` PEFT prefix rather than by `target_modules`, so this did not matter.
  - Kept the merged product in float32 and cast to float16 only at the end; `.contiguous()` on every tensor before `save_file`.
- Anything in the task text or documentation that was unclear:
  - TASK.md says `embed_in.weight` and `embed_out.weight` (206 MB each) are "larger than" the 512 MiB budget and must be stored alone; they are not larger, so I applied the stated budget rule (greedy packing in key order, at most 512 MiB of tensor data per shard). This gives 4 shards; the embeddings share shards with other tensors.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
