# T3 self-report (condition B: BrainSurgery plan)

- Final artifact path: `out/T3/` (4 shards `model-0000N-of-00004.safetensors`
  plus `model.safetensors.index.json`); plan at `out/T3/plan.yaml`.

- Number of times you executed the script or plan: 3 total `brainsurgery`
  invocations — 1 read-only dump (`out/T3/inspect.yaml`, since deleted) to
  confirm tensor names, and 2 executions of `out/T3/plan.yaml`.

- Which executions failed, and why (one line each):
  - plan run 1: `assert: { count: { of: 'h\.\d+\.attn\.bias', is: 0 } }` failed
    with `count.of matched zero tensors` — a reference that matches nothing is a
    resolution error before `count` compares, so `count ... is: 0` can never
    hold; replaced with `not: { exists: ... }`. Nothing was written.
  - plan run 2: succeeded, wrote `out/T3/`.

- Pitfalls or surprises you hit (one line each):
  - Zero-match references are an error, not an empty match set, so "assert this
    is gone" must be `not: { exists: ... }`, never `count: is: 0`.
  - "Exactly 48 tensors are bfloat16" is not directly expressible; I encoded it
    as count==48 on the projection pattern, `dtype: bfloat16` on that pattern,
    and `dtype: float32` on its complement via a negative-lookahead full-match
    regex — together those pin the dtype of all 148 tensors.
  - `cast_` (in-place) is the right tool here; `cast` creates new destinations
    and would have changed tensor names.
  - Shard units are binary, so `shard: 64MB` is exactly the 67,108,864-byte
    budget the task asks for; `wte.weight` (154 MB) lands alone in its own shard
    automatically, as the README documents.
  - Doing the `delete` before the final asserts matters: `count: { of: '.*', is:
    148 }` is only meaningful once the 12 buffers are gone.

- Anything in the task text or documentation that was unclear:
  - The README does not say that a zero-match reference raises rather than
    yielding an empty set; that cost the one failed run.
  - `assert.dtype` help says "the tensor has the given dtype" (singular) without
    stating that a multi-match reference requires *every* match to have it —
    behaviour I relied on and confirmed empirically.

- Tools used (condition F): n/a (condition B).

- Approximate time spent, if you can tell: ~10 minutes, most of it reading
  `docpack/help.txt` for the assert operators.
