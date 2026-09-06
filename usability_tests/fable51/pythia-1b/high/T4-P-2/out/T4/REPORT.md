# T4 participant self-report (condition P)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion in my own post-write sanity check. I asserted every written tensor is F16, but the base contains non-float16 mask buffers (`gpt_neox.layers.<i>.attention.bias`, dtype U8/bool). The merge and the file write were already correct; only the extra check was too strict. Fixed by comparing each written tensor's shape/dtype against the base tensor instead of assuming F16.
- Pitfalls or surprises you hit (one line each):
  - The task text says the checkpoint is "float16", but the `attention.bias` mask buffers (and `masked_bias`) are not float16; a blanket dtype assumption fails.
  - `attention.bias` in GPT-NeoX is a mask buffer, not an MLP bias; the anchored regex on `mlp.dense_h_to_4h|dense_4h_to_h` avoids overmatching it.
  - Task vectors must be taken against the original base tensor (both `ft1 - base` and `ft2 - base` computed from the same float32 copy of base), not against the partially merged result.
- Anything in the task text or documentation that was unclear:
  - "Every tensor is float16" is only true for the parameter tensors; the mask buffers are a different dtype. Not a blocker, but it caused my one failed execution.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes, two ~10-15 s runs.
