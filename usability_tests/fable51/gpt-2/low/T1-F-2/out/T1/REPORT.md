# T1 participant self-report

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The "Required checks" mention blocks 9, 10, 11; those are the stale post-shift indices, not the dropped blocks (2, 5, 8). The script checks both: no index >= 9 remains, and blocks are exactly 0..8.
  - Building a fresh dict from a fixed old->new map avoids in-place rename collisions entirely; a collision guard is still asserted.
- Anything in the task text or documentation that was unclear: the "blocks 9, 10, 11" wording in the checks reads oddly next to "remove blocks 2, 5, 8" until one realizes it targets leftover old indices.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file`/`save_file` for I/O.
  - `torch` 2.14.0: tensor container only (no ops).
  - Python `re` for the `^h\.(\d+)\.(.+)$` block pattern. No mergekit or torch-state-bridge: a 40-line script with explicit checks was simpler than configuring a tool and then still writing the checks.
- Approximate time spent, if you can tell: about 2 minutes.
