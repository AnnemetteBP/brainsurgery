# Participant self-report: T3 (Pythia-1B, condition P)

- Final artifact path: `out/T3/solution.py` (output: 9 shard files plus `out/T3/model.safetensors.index.json`, 196 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Anchored the projection regex to the four exact module names so it cannot hit embeddings, norms or biases.
  - Matched the three buffer names with an anchored regex; `attention.bias` must not be confused with projection biases like `attention.dense.bias`.
  - The two embeddings are 412 MB each once upcast to float32 (TASK.md quotes 206 MB, their float16 size), so they exceed the 256 MiB budget and each go alone in a shard.
  - Greedy sharding in key order (safetensors loads keys sorted) keeps each multi-tensor shard under the budget.
- Anything in the task text or documentation that was unclear:
  - The "206 MB each" figure refers to the input float16 size; after the required float32 upcast the embeddings are twice that. Either way they are stored alone.
  - Shard file naming was not specified; used the HF convention `model-0000i-of-0000n.safetensors`.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes
