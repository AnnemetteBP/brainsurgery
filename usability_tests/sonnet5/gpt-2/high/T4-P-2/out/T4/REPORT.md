# Participant self-report

- Final artifact path: `out/T4/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None specific to this task; the main care point was doing the shared-tensor
    verification (names, shapes, dtypes, bit-exact equality) against the
    unmodified base for both fine-tunes before computing anything, and taking
    each task vector (`ft - base`) against that same unmodified base rather
    than against an already-merged tensor.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes (single pass, no debugging needed).
