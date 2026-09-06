## Participant self-report

- Final artifact path: `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none — base checkpoint is
  sharded (index.json + 2 files) while ft1/ft2 are single-file, so loading
  needed two small helper functions, but otherwise straightforward.
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): name, version, and why: `torch` 2.14.0 and
  `safetensors` 0.5.3 only, via a plain script (`solution.py`). Chose this
  over `mergekit`'s task-arithmetic YAML config because the task's precise
  precondition check (identical non-MLP tensors across all three checkpoints,
  bit-exact) and exact-count assertions (48 merged, 114 total) are easiest to
  express and guarantee directly in a short script rather than through a
  merge-tool config that doesn't expose those checks natively.
- Approximate time spent, if you can tell: ~5 minutes
