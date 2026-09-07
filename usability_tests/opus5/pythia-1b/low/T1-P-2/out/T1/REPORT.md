# T1 self-report (condition P)

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: avoided by building a fresh output dict keyed by the new names instead of renaming in place, plus an explicit collision assert.
  - safetensors rejects shared storage, so values are cloned before saving.
- Anything in the task text or documentation that was unclear: no; the QKV layout detail is irrelevant for this pure-rename task.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~3 minutes
