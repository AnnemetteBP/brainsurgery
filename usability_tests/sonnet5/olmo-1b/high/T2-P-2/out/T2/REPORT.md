## Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each): none of note — head boundaries (128-dim heads, head 5 at rows/cols 640..767) and the row-block (q/k/v) vs column-block (o_proj) layout were as described in TASK.md.
- Anything in the task text or documentation that was unclear: no, the task text fully specified the tensor layout, the rows/columns to keep, and the required checks.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes (single script write + single run).
