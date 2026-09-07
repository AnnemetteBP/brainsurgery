# T4 Participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Task vectors must be computed from the untouched base for both fine-tunes; computed both deltas from the same float32 base copy before adding.
  - Called `.contiguous()` on tensors before `save_file` to avoid safetensors layout errors; float16 base tensors were already contiguous.
- Anything in the task text or documentation that was unclear: nothing; the MLP name pattern and layer range (0..15) were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
