## Participant self-report

- Final artifact path: `out/T1/solution.py` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to compute the old-index -> new-index mapping programmatically (enumerate surviving indices in order) rather than hardcode it, to avoid a transcription error against the 12-entry mapping in the spec.
  - Renumbering had to happen via a fresh dict of new names rather than in-place mutation, to avoid any chance of a new name colliding with an old name still pending processing (e.g. old 3 -> new 2 could otherwise clash with an unprocessed old block 2 if blocks weren't dropped first).
- Anything in the task text or documentation that was unclear: none; the explicit old->new index list in the task removed any ambiguity about ordering.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
