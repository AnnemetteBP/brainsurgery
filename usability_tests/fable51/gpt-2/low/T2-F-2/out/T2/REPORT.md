# T2 self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `attn.bias` is the causal mask buffer, not a per-head bias; matched only `attn.c_attn.*` and `attn.c_proj.weight` by suffix so `mlp.c_proj.weight` and the mask stay untouched.
  - Fancy-indexed tensors are non-contiguous views; called `.contiguous()` before `save_file`.
- Anything in the task text or documentation that was unclear: nothing; the explicit column ranges made the layout unambiguous.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: index-select the kept head blocks.
  - `safetensors` 0.5.3: load/save the checkpoint.
  - Did not use transformers `prune_heads`: it would need a config edit and a model load/save round trip, and its output layout is harder to keep bit-exact to the spec's column order than direct slicing.
- Approximate time spent, if you can tell: about 2 minutes.
