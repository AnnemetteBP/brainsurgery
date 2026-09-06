## Participant self-report

- Final artifact path: `out/T1/plan.yaml` (output written to `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The main hazard is ordering: renumbering must run in ascending old-index
    order (3→2, 4→3, 5→4, 7→5, 8→6, 9→7, 11→8, 12→9, 13→10, 15→11) with the
    four block deletes done first, so every `move` destination slot is
    already empty (freed by a prior delete or by the immediately preceding
    move) when it is written to — `move` refuses to overwrite an existing
    destination, so the wrong order fails loudly rather than silently
    corrupting a layer.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): n/a (condition B, plan only).
- Approximate time spent, if you can tell: a few minutes (mostly reading
  `docpack/README.md` and `help.txt` for the exact `move`/`delete`/`assert`
  key names before writing the plan).
