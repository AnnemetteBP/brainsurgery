# Participant self-report

- Final artifact path: `out/T3/model.safetensors.index.json` and the four `out/T3/model-0000*-of-00004.safetensors` shard files
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The 154,389,504-byte `wte.weight` tensor exceeds the shard limit and therefore had to be isolated in its own allowed oversized shard.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): PyTorch 2.14.0 for exact dtype conversion and tensor checks; safetensors 0.5.3 for loading and writing the checkpoint; Python standard-library `json` for the index.
- Approximate time spent, if you can tell: About 5 minutes.
