# Participant self-report: T1 (GPT-2 124M, condition P)

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`, 121 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering into a fresh dict (rather than in place) avoids the shift-order collision hazard entirely; a collision check guards it anyway.
  - The block regex is anchored (`^h\.(\d+)\.`) so `attn.bias` mask buffers and `mlp.c_proj` are handled by block index only, never by suffix.
- Anything in the task text or documentation that was unclear: The check "no tensor of blocks 9, 10, 11 remains" refers to post-renumbering indices; I also verified indices are exactly 0..8 and that each survivor maps to the same storage as its source.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
