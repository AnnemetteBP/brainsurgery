## Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None functionally, but it's easy to misread the fused QKV layout as
    `[q-block | k-block | v-block]` segments instead of GPT-NeoX's
    per-head-interleaved layout (768-row blocks each containing that head's
    q, k, v); the task text calls this out explicitly, so I sliced by
    768-row head blocks rather than by q/k/v segments.
  - The row axis for query_key_value and the column axis for dense both
    encode "head", but at different granularities (768 rows vs 256 columns
    per head) since dense only sees the per-head output (256-dim), not the
    fused q/k/v (768-dim); worth double-checking against the given target
    shapes before writing.
- Anything in the task text or documentation that was unclear: no, the
  required row/column ranges to keep were given explicitly per tensor, which
  made verifying the slicing straightforward.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes to write and verify
  the script.
