# T1 report

## Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The renumbering moves must be ordered so each destination block index is
    already vacated (by a prior `delete` or an earlier `move` in the same
    list) before it is written to, otherwise `move` would fail with a
    "destination already exists" error; sequencing the seven moves in
    ascending target order (3→2, 4→3, 6→4, 7→5, 9→6, 10→7, 11→8) makes every
    destination free at the time it is used.
  - `move`'s `to` field supports regex capture-group rewriting exactly like
    `copy`/`equal` (`\1`), so one `move` per block (e.g.
    `from: 'h\.3\.(.*)', to: 'h.2.\1'`) renames all 13 tensors of that block
    in a single transform instead of listing each tensor name.
- Anything in the task text or documentation that was unclear: none; the
  README's note under `assert.equal` ("`right` is resolved as a rewrite of
  each `left` match, exactly like `to` in `copy`/`move`") was the key detail
  needed to confirm `move` accepts regex/capture-group batch renames, since
  the `move`-specific help text only shows single-tensor examples.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: about 10 minutes, including
  reading the doc pack and verifying the output bit-exact against the input.
