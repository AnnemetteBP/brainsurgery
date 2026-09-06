# Participant self-report: T4 (condition P)

- Final artifact path: `out/T4/solution.py` (output `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Task vectors must both be taken against the original base tensor, not the running merge; computed both deltas from `base` in one expression.
  - Used an anchored regex on `h.<i>.mlp.(c_fc|c_proj).(weight|bias)` so the layer-count and tensor-count checks catch overmatch.
- Anything in the task text or documentation that was unclear: nothing
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
