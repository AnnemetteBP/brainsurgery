# Participant self-report

- Final artifact path: `out/T2/plan.yaml` (plan), `out/T2/model.safetensors` (output)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `concat`'s `from` list requires each entry to resolve to exactly one tensor, so the
    per-layer slice-and-rebuild had to be a separate `concat` per layer (48 total)
    rather than one pattern-matched transform across all 16 layers.
  - `move`/`delete`/`assign`/`copy` all support regex capture-group rewrites across many
    tensors in one call (confirmed by the `equal` assert docs, which say `right` rewrites
    "like `to` in copy/move"), so the rename-back-to-original-name step and the
    delete-of-originals step could each be done in one transform for all 16 layers.
  - Regex tensor references are full-match, so a pattern for the original tensor name
    (e.g. `...query_key_value\.weight`) does not accidentally match the temporary
    `...weight__pruned` name; no extra anchoring needed.
- Anything in the task text or documentation that was unclear: none; the interleaved
  per-head row layout and column-block layout for `dense.weight` were stated precisely
  enough to compute the exact keep-ranges (`0:3840`+`4608:6144` for rows,
  `0:1280`+`1536:2048` for columns) directly from head index 5, without trial and error.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~15 minutes, most of it reading the `concat`/
  `move`/`delete` help entries to confirm which transforms allow multi-tensor pattern
  matching versus single-tensor references.
