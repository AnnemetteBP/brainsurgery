# Participant self-report

- Final artifact path: `out/T5/solution.py` (checkpoint at `out/T5/model.safetensors.index.json` and six shard files)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The two embedding tensors had to be isolated even though each is below the nominal 512 MiB shard limit; merge arithmetic also had to remain float32 until the final float16 cast.
- Anything in the task text or documentation that was unclear: The statement that the roughly 206 MB embedding tensors are individually larger than 512 MiB is inconsistent, but the explicit instruction to store each alone was clear and was followed.
- Tools used (condition F): N/A (condition P).
- Approximate time spent, if you can tell: About 5 minutes.
