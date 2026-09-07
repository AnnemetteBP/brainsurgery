# Participant self-report

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering into a fresh dict (rather than renaming in place) avoids the collision hazard entirely; added an explicit collision check anyway.
  - Input is sharded; loaded both shards via the index file and merged them.
- Anything in the task text or documentation that was unclear:
  - The check "no tensor of blocks 12, 13, 14, 15 remains" refers to post-renumbering indices, which is implied but not stated.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: about 2 minutes
