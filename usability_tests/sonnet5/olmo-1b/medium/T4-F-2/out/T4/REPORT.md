# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`.
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Inputs mix a sharded base (`model.safetensors.index.json` + two shards) with
    single-file fine-tunes (`model.safetensors`); the loader has to handle both
    layouts to build a uniform `{tensor_name: shard_path}` map.
  - `torch.equal` bit-exact comparison (not an approximate `allclose`) is what
    the spec means by "identical" for the shared-tensor precondition, so I used
    that rather than a tolerance-based check.
- Anything in the task text or documentation that was unclear: no — the formula,
  MLP tensor list, and required checks were all fully specified.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, via a
  plain script. I did not reach for `mergekit`'s task-arithmetic merge method
  because this task's precondition check (verify all non-MLP tensors are
  bit-identical across three checkpoints, abort otherwise) and the exact
  ordering requirement (each task vector taken against the unmodified base)
  are easiest to state and audit directly against loaded tensors; a
  hand-written script keeps the abort conditions and the arithmetic in one
  place I can read end to end, and avoids translating those preconditions
  into a YAML merge config with a separate verification step bolted on.
- Approximate time spent, if you can tell: a few minutes (read task, write
  script, run once, verify output against a manual reconstruction of one
  tensor).
