# Participant self-report: T3 (Pythia-1B, condition P)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 9 shard files `model-0000N-of-00009.safetensors` plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - The task text quotes the embeddings at 206 MB each (their float16 size); after the required upcast to float32 they are 412 MB, still over the shard budget, so each is stored alone in its own shard as specified.
  - Anchored regexes were used for both the projection set and the buffer set so that `attention.bias` (a buffer) is never confused with a projection bias, and neither embeddings nor norm weights match the `.weight` projection pattern.
- Anything in the task text or documentation that was unclear:
  - The sharding packing strategy (greedy in key order vs. some other grouping) and shard file naming are not specified; I used HF-style greedy packing in the source key order with `model-XXXXX-of-XXXXX.safetensors` names and an index with `metadata.total_size` and `weight_map`.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: a few minutes.
