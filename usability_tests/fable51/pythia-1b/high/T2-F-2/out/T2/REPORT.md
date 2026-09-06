# T2 self-report (Pythia-1B, condition F)

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Column-sliced `dense.weight` is non-contiguous after narrowing; assembled with `torch.cat(...).contiguous()` so `save_file` accepts it.
  - `attention.masked_bias` is a 0-d tensor and `attention.bias` is uint8, so the untouched-tensor byte comparison had to tolerate scalars and non-float dtypes.
  - Used exact tensor names (a set built from the 16 layer indices) instead of a regex to avoid overmatching `mlp.*` or `dense.bias`.
- Anything in the task text or documentation that was unclear:
  - "Loadable as the same architecture with 7 heads per layer" does not hold for stock HF `GPTNeoXConfig` (7 heads x 256 = 1792 != hidden_size 2048), so I did not attempt a `transformers` load test; the required result is the checkpoint slicing spec, which I followed literally.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: lazy per-tensor read of the input and `save_file` for the output, preserving dtype bit-exactly.
  - `torch` 2.14.0: `narrow`/`cat` for slicing and `torch.equal` for the verification passes.
  - Did not use `transformers.prune_heads`: it operates on a loaded model and would require a config with mismatched head geometry, and it would not guarantee the exact row ordering the spec dictates; direct slicing is smaller and verifiable.
- Approximate time spent, if you can tell: about 3 minutes, including the ~5 s run.
