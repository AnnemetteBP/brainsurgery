# Participant self-report: T5 (Pythia-1B, condition P)

- Final artifact path: `out/T5/solution.py` (output shards + `model.safetensors.index.json` in `out/T5/`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each): none failed; execution 2 was a re-run after changing the sharding so each embedding matrix sits alone in its own shard.
- Pitfalls or surprises you hit (one line each):
  - TASK.md says the 206 MB embeddings are "larger than" the 512 MiB budget and stored alone; they are not larger, so I isolate tensors above 128 MiB to satisfy both readings while staying under the budget.
  - `adapter_config.json` lists `target_modules = ["query_key_value"]`, not `attention.query_key_value` as TASK.md says; I derived targets from the adapter tensor names instead of the config.
- Anything in the task text or documentation that was unclear: the embedding-alone sharding rule vs. the 512 MiB budget (see above).
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes
