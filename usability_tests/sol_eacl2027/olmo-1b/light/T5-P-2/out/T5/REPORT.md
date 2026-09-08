# Participant self-report

- Final artifact path: `out/T5/solution.py` (checkpoint and index in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The 5.1 GB base needed to be streamed and repacked shard-by-shard instead of loading the complete checkpoint into memory.
- Anything in the task text or documentation that was unclear: The statement that the 412 MB embedding and LM-head tensors are larger than the 512 MiB limit is numerically inconsistent; I applied the stated byte limit exactly, so each may share a shard only when the shard remains within 536,870,912 bytes.
- Tools used (condition F): Not applicable (condition P).
- Approximate time spent, if you can tell: About 4 minutes.
