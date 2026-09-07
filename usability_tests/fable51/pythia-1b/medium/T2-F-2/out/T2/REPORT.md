# T2 self-report (Pythia-1B, condition F)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None. The task text fully specifies the interleaved 768-row-per-head layout of `query_key_value`
    and the 256-wide column blocks of `dense.weight`, so a direct slice was unambiguous.
  - Deliberately avoided `transformers` `prune_heads`: it would reshape modules and go through
    `save_pretrained`, which is riskier for bit-exactness and key/dtype preservation than slicing
    the state dict directly.
- Anything in the task text or documentation that was unclear: nothing.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file` / `save_file` for checkpoint I/O (float16 preserved, single file).
  - `torch` 2.14.0: `narrow` + `cat` to drop the head-5 block on the right axis of each tensor.
  - No mergekit / peft / torch-state-bridge: none of them expresses intra-tensor block removal.
- Approximate time spent, if you can tell: about 3 minutes.

Checks enforced before writing: layer-0 qkv weight `[5376, 2048]`, qkv bias `[5376]`,
dense weight `[2048, 1792]`, exactly 244 tensors, plus unchanged key set, dtypes, and
identity of every non-head-bearing tensor. A separate verification pass confirmed all 48
resliced tensors bit-match the explicit row/column ranges in TASK.md and the other 196
are unchanged.
