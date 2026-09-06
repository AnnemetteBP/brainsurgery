# Participant self-report

- Final artifact path: `out/T3/solution.py` (output: 4 shard files plus `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `h.<i>.attn.bias` is a causal-mask buffer, not a projection bias, so it had to be matched by exact name and dropped rather than kept as float32.
  - Projection matrices were matched with an anchored regex on the four module names to avoid `.*weight` overreach onto embeddings and layer norms.
  - `wte.weight` (154 MB) exceeds the 64 MiB budget, so the packer allows an oversized single tensor to occupy its own shard.
- Anything in the task text or documentation that was unclear: shard file naming and the index `metadata` field are unspecified; used the HuggingFace `model-0000i-of-0000n.safetensors` convention with `total_size`.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: about 2 minutes
