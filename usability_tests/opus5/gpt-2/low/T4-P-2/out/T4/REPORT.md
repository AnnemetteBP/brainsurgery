# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The ordering hazard is real: both task vectors must be computed from the original `base` tensor, so I build a fresh output dict instead of updating the base state dict in place.
  - `inputs/` also contains a `lora/` directory that is irrelevant to this task; ignored it.
  - safetensors rejects non-contiguous/shared storage, so every output tensor is cloned and made contiguous.
- Anything in the task text or documentation that was unclear:
  - Step 1 says "every tensor outside the 48 MLP tensors is identical"; I read that as bit-exact equality (`torch.equal`) rather than an allclose tolerance.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes.
