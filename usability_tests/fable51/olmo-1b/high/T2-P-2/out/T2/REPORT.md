# T2 self-report (condition P, OLMo-1B-0724-hf)

- Final artifact path: `out/T2/model.safetensors` (script: `out/T2/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Input is sharded (two files plus index); the script loads both shards via the index's `weight_map` and checks for duplicate or missing keys.
  - `index_select` on a fresh index tensor keeps the kept blocks in original order; sliced tensors are made contiguous before saving so safetensors accepts them.
  - Matched tensor names by exact `self_attn.<q|k|v|o>_proj.weight` leaf rather than a loose substring, so MLP and norm tensors cannot be touched by accident.
- Anything in the task text or documentation that was unclear: nothing material. The task did not say whether to carry safetensors metadata; the output uses `{"format": "pt"}`, which is what the HF loader expects.
- Tools used (condition F): not applicable (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: about 2 minutes.
