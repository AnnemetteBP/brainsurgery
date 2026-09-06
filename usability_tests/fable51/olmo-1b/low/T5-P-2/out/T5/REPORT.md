# Participant self-report: T5 (condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`, 10 shards + index)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - PEFT adapter names carry a `base_model.model.` prefix that must be stripped to map onto `model.layers.<i>...weight`.
  - The 512 MiB shard budget is on tensor data only; shard files end up slightly larger than 512 MiB because of the header.
  - The two 412 MB embedding/lm_head tensors each land alone in a shard by the greedy packing rule.
- Anything in the task text or documentation that was unclear: nothing significant; `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj` while TASK.md says `self_attn.q_proj`, but the tensor names resolve the same.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
