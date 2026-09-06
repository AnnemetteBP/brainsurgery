## Participant self-report

- Final artifact path: `out/T4/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None; had to be careful to compute both task vectors against the
    unmodified `base` (not against each other or a partially-merged result),
    and to do the arithmetic in float32 before casting back to float16.
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes
