# Participant self-report: T3 (GPT-2 124M), condition P

- Final artifact path: `out/T3/solution.py` (outputs: `out/T3/model-0000{1..4}-of-00004.safetensors`, `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all checks and wrote 4 shards (59 + 58 + 30 tensors under the 64 MiB budget, plus `wte.weight` alone in the fourth).
- Pitfalls or surprises you hit (one line each):
  - Targeting by an anchored regex on the four exact projection names, not `.*weight`, so embeddings and layer norms stay float32.
  - `h.<i>.attn.bias` looks like a parameter bias but is the causal-mask buffer; matched it by exact name and dropped it before the other checks.
  - Greedy sharding in key order would have overflowed if `wte.weight` were placed in a mixed shard, so any tensor larger than the budget is emitted alone.
- Anything in the task text or documentation that was unclear: the shard file naming scheme and whether `weight_map` needs a `metadata.total_size` entry were unspecified; I used the HF convention (`model-XXXXX-of-XXXXX.safetensors` plus `metadata.total_size`).
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: about 2 minutes.
