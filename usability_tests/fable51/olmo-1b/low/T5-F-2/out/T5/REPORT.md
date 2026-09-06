# T5 participant self-report

- Final artifact path: `out/T5/solution.py` (output: `out/T5/model-0000N-of-00010.safetensors` + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1 did not fail (all checks passed), but its greedy sharder paired `model.embed_tokens.weight` with a 64 MiB MLP tensor, while the task says the two 412 MB tensors are stored alone; I changed the rule and re-ran.
- Pitfalls or surprises you hit (one line each):
  - The 412 MB tensors are below the 512 MiB budget, so "stored alone" is not implied by the budget; I added a rule that any tensor over half the budget gets its own shard.
  - PEFT adapter keys carry the `base_model.model.` prefix that must be stripped to reach base names.
- Anything in the task text or documentation that was unclear:
  - "A single tensor larger than that is stored alone" vs. the two named tensors being smaller than the budget; the exact reference sharding algorithm (shard order, fill rule) is not specified.
- Tools used (condition F): torch 2.14.0 (matmul, tensor ops), safetensors 0.5.3 (`safe_open`, `save_file`). Chosen over peft `merge_and_unload` because it avoids instantiating the model, keeps unchanged tensors bit-exact, and gives full control over sharding.
- Approximate time spent, if you can tell: about 3 minutes.
