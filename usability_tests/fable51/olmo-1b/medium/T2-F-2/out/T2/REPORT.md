# T2 participant self-report (condition F)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None in the run itself. The task text fully specifies the layout (Linear `[out, in]`, q/k/v heads as row blocks, o_proj heads as column blocks), so no layout discovery was needed.
  - Used `index_select(...).contiguous()` so safetensors does not reject non-contiguous views on save.
- Anything in the task text or documentation that was unclear: nothing. One note: the input is sharded, so the solution must merge two shards into one output file; the task states this clearly.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file` of both shards and `save_file` of the single output file.
  - `torch` 2.14.0: `index_select` on the head axis of each projection, plus equality checks.
  - Plain Python script instead of `transformers.prune_heads`: the spec asks for a bit-exact checkpoint with unchanged tensor names and a single safetensors file; `prune_heads` would require a model load, config edits and `save_pretrained`, adding more places for a mismatch with no gain for a task this fully specified.
- Approximate time spent, if you can tell: about 2 minutes.

Checks enforced by the script before writing: shapes of layer-0 q/k/v/o projections, all 16 layers, exact 114-tensor count, float32 dtype preserved, and exactly 64 tensors touched. After writing, the file is re-read and key set and count re-verified. A separate one-off check confirmed every output tensor is bit-exact against slicing the input directly.
