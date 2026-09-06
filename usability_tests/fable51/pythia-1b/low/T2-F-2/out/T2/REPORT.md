# T2 self-report (condition F)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `failed_assertion` — my own extra sanity check "all tensors are float16" fired because the `attention.bias` causal-mask buffer is uint8. Not a required check; removed it. Nothing was written (assertions run before save).
- Pitfalls or surprises you hit (one line each):
  - The checkpoint is not uniformly float16: `gpt_neox.layers.<i>.attention.bias` is U8, so a blanket dtype assertion is wrong; per-tensor dtypes were preserved untouched instead.
  - Slices must be made `.contiguous()` before `save_file`, done via `torch.cat`.
- Anything in the task text or documentation that was unclear: Nothing; the row/column ranges were explicit. The "stored in float16" wording slightly oversells uniformity given the U8 mask buffer.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load/save the checkpoint.
  - `torch` 2.14.0: `torch.cat` slicing of head blocks.
  - Did not use transformers `prune_heads`: it renumbers nothing here but would require loading the model, and output must be a single 244-tensor safetensors file with the original names, which a plain script guarantees more directly.
- Approximate time spent, if you can tell: about 2 minutes.
