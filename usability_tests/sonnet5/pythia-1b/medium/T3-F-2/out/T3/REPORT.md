# Participant self-report

- Final artifact path: `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Upcasting the two embedding matrices from float16 to float32 doubles
    their size to ~412 MB each, above the 256 MiB shard budget — confirmed
    this matches the task's explicit exception (single oversized tensor
    gets its own shard) rather than treating it as a bug.
  - Buffer names overlap textually with parameter names
    (`attention.bias` vs. `attention.dense.weight`) — used anchored regexes
    for both the bf16-projection pattern and the buffer-drop pattern so
    neither could accidentally overmatch the other.
- Anything in the task text or documentation that was unclear: none; the
  per-layer tensor names, shapes, and shard-budget exception were stated
  precisely enough to write assertions directly from the spec.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, used
  directly in a ~140-line script (`load_and_transform` casts by exact regex
  match on the 4 named projection families and drops the 3 named buffer
  patterns, `run_required_checks` asserts the 4 required conditions before
  any writing, `plan_shards`/`write_sharded` greedily bin-pack tensors into
  256 MiB shards with a single-tensor exception, writing shard files plus a
  `model.safetensors.index.json`). Chose a plain script over mergekit or
  `transformers` `save_pretrained(dtype=...)` because this task needs
  per-tensor dtype targeting by exact name (not a single global dtype), buffer
  deletion, and a specific shard-budget/exception rule — a short script gives
  full control of all three without fighting a merge-config or HF save API
  that assumes one dtype for the whole checkpoint.
- Approximate time spent, if you can tell: ~10 minutes.
