# Participant self-report

- Final artifact path: `out/T2/plan.yaml` (writes `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `concat`/`move`/`delete` all require the destination name to not already
    exist, so pruning in place needs a temp `__pruned` name, deleting the
    original, then `move`-ing the temp tensor back onto the original name.
  - Row slices for q/k/v use `[start:end, :]`; the o_proj slice is on the
    column axis, `[:, start:end]`, since it consumes heads as input columns.
- Anything in the task text or documentation that was unclear: none; the
  per-tensor row/column block description in TASK.md was sufficient to derive
  the exact slice bounds (head 5 spans rows/cols 640..767 of a 2048-wide,
  16x128 layout).
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: one pass, plan written and run once.
