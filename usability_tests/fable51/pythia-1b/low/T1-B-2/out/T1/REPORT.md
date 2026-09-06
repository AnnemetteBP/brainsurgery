# T1 self-report (condition B, Pythia-1B)

- Final artifact path: `out/T1/plan.yaml` (output checkpoint `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 2 (the first run succeeded; the second was an identical re-run only to capture the exit code, since the first run's output was truncated by `tail`)
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `move` requires the destination to be free, so the renumbering moves must be ordered by ascending old index (3->2 before 4->3, etc.); descending order would collide with surviving blocks.
  - The `help.txt` grep by command name needs the box-drawing "Help for <cmd>" heading; plain `^move` does not match.
- Anything in the task text or documentation that was unclear:
  - The README documents `\1` rewrites only under `assert: equal` ("exactly like `to` in copy/move"); the `move` help itself shows no regex example, so I inferred capture-group support from that cross-reference.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
