# T2 self-report (condition F, OLMo-1B-0724-hf)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`, 114 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Input is sharded; merged both shards via `model.safetensors.index.json` into one state dict before slicing.
  - Slices from `index_select` are made contiguous before `save_file` to avoid safetensors layout errors.
- Anything in the task text or documentation that was unclear: nothing; row/column ranges were explicit.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: `index_select` on head-block indices (rows for q/k/v, columns for o_proj).
  - `safetensors` 0.5.3: `load_file` / `save_file` for bit-exact float32 I/O.
  - Did not use `transformers.prune_heads`: it would require loading the full model and its head-pruning path can re-derive tensors in ways that risk non-bit-exact output; a direct slice is simpler and exactly matches the spec.
- Approximate time spent, if you can tell: about 2 minutes.

Checks enforced in `solution.py` before writing: layer-0 q/k/v shapes `[1920, 2048]`, o_proj `[2048, 1920]`, tensor count 114, plus all-layer shapes and float32 dtype. A separate post-run verification compared every output tensor bit-exactly against the expected slice of the input.
