# Participant self-report: T2 (Pythia-1B head pruning), condition P

- Final artifact path: `out/T2/solution.py` (output checkpoint `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The GPT-NeoX fused qkv is interleaved per head (768-row block of q, k, v per head), so pruning one head is a single contiguous row-block removal rather than three separate slices; the task text stated this clearly.
  - `torch.cat` of two narrowed views is used and made `.contiguous()` before saving, since safetensors rejects non-contiguous tensors.
- Anything in the task text or documentation that was unclear: nothing; row and column ranges were given explicitly and matched the derived head boundaries (768*5 = 3840, 256*5 = 1280).
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
