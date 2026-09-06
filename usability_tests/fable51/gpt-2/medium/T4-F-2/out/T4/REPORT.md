# T4 participant self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None. The MLP name pattern is anchored (`^h\.\d+\.mlp\.(c_fc|c_proj)\.(weight|bias)$`) so it cannot overmatch attention or layernorm tensors; the script asserts it matches exactly 48 names.
  - Task vectors are computed each against the untouched `base` dict, never against the accumulating output, so ordering is not an issue.
- Anything in the task text or documentation that was unclear: nothing.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load/save of the three checkpoints and the output.
  - `torch` 2.14.0: `torch.equal` for the bit-exact shared-tensor check and float32 arithmetic for the merge.
  - I did not use `mergekit` task arithmetic: it does not perform the required precondition check that all non-MLP tensors are identical across the three checkpoints, and it would apply the merge to every tensor rather than only the 48 MLP tensors, so a plain script was both simpler and safer.
- Approximate time spent, if you can tell: about 2 minutes.
