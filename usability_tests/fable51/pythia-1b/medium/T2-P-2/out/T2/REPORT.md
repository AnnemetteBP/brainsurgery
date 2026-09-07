# Participant self-report: T2 (Pythia-1B, condition P)

- Final artifact path: `out/T2/solution.py` (output checkpoint at `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None. The task text fully specified the GPT-NeoX interleaved per-head layout (768-row blocks of q/k/v per head) and the exact row/column ranges to keep, so no layout discovery was needed.
  - Slices from `narrow` + `cat` were made contiguous before saving to avoid safetensors rejecting non-contiguous tensors.
- Anything in the task text or documentation that was unclear: nothing; the required ranges and shapes were explicit and internally consistent (8 heads x 768 = 6144, 8 x 256 = 2048).
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3 only).
- Approximate time spent, if you can tell: about 2 minutes.
