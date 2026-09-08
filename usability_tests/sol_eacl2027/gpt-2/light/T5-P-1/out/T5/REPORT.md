# Participant self-report

- Final artifact path: `out/T5/` (`model.safetensors.index.json` and five shard files)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The oversized `wte.weight` tensor needed to be isolated while all ordinary shards remained at or below 100 MiB.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): Not applicable (condition P); used only Python, PyTorch, and safetensors.
- Approximate time spent, if you can tell: About 5 minutes.
