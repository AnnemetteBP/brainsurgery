# T1 self-report

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: avoided by building a fresh output dict keyed by the new names instead of renaming in place, plus an explicit collision check.
  - `attn.bias` is a mask buffer, not a bias of a projection; it is part of the 13 per-block tensors and must move with the block.
  - Block matching is anchored on `^h\.(\d+)\.` so `mlp.c_proj` and other suffixes are untouched.
- Anything in the task text or documentation that was unclear: nothing; the required key set and count were fully specified.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: a few minutes.
