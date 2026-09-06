# T1-F-2 Participant self-report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None specific to this task; renumbering was done via a single `enumerate()` over the surviving old indices in ascending order, so there's no risk of a partially-renumbered block colliding with a not-yet-moved one (all tensors are read from the original dict and written into a fresh dict keyed by new indices, never mutated in place).
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): `safetensors` 0.5.3 (`safe_open`/`save_file`) for checkpoint I/O, plain Python (`re`) for the key rewrite. Chose a direct script over `mergekit` layer-slicing or `torch-state-bridge` because the transform is a flat rename+drop on a single file with no cross-checkpoint merging or arithmetic involved — a plain regex-driven rewrite is the smallest tool that can express "select by block index, drop 4, remap the rest to a dense 0..11 range" and makes the collision-avoidance property (build a new dict, never overwrite in place) directly inspectable.
- Approximate time spent, if you can tell: a few minutes (single script, single successful run).
