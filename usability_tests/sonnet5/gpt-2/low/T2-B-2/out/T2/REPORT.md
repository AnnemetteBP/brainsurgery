# Participant self-report

- Final artifact path: `out/T2/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat`'s destination must not already exist, so each head-bearing tensor
    is rebuilt under a temporary `_pruned` name, the original is `delete`d,
    then the pruned tensor is `move`d back to the original name.
  - Had to double-check the column/row block boundaries by hand (each 768-wide
    q/k/v segment splits into 12 heads of 64 columns, head 5 is columns
    320:384 within each segment) before writing the six-piece keep list.
- Anything in the task text or documentation that was unclear: none; the
  per-layer keep ranges were given explicitly in TASK.md, and `concat`/`delete`/
  `move`/`assert` from `docpack/help.txt` covered everything needed.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes.
