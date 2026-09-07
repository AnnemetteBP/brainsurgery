# Participant self-report: T3 (condition P)

- Final artifact path: `out/T3/solution.py` (output checkpoint: `out/T3/model-*-of-00010.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion. My own shard-budget sanity check fired on the first shard, which holds only `model.embed_tokens.weight` (412 MB); the task allows a single oversized tensor to be alone in its shard, so the check needed the `len(names) > 1` qualifier. Failed before any files were written.
- Pitfalls or surprises you hit (one line each):
  - The "at most 256 MiB per shard" rule has an exception for single oversized tensors; a naive post-write check contradicts it.
  - Used an anchored regex on `self_attn.[qkvo]_proj` / `mlp.(gate|up|down)_proj` `.weight` and cross-checked the match count against 112, so embeddings and lm_head could not be swept in.
- Anything in the task text or documentation that was unclear:
  - Shard naming and the index `metadata.total_size` field are not specified; I used the HF convention `model-XXXXX-of-NNNNN.safetensors` and greedy packing in original key order.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
