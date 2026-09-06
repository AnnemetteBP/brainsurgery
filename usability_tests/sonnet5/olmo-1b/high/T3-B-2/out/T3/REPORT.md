## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts on the first try.
- Pitfalls or surprises you hit (one line each):
  - `dtype`/`count` asserts take a full-match regex `of` reference that can match many tensors at once, so a negative-lookahead regex (`(?!model\.layers\.\d+\...\.weight$).+`) could be used to assert "everything else stays float32" without listing `embed_tokens`/`lm_head` by name.
  - `cast_` (in-place) was simpler than `cast` + `delete`/`move` here since tensor names must stay the same; it also keeps the whole run on a single alias (`model`), which output-alias inference requires.
  - The 256 MiB shard budget only counts tensor data, not headers, so shard file sizes on disk are a few KB larger than 268,435,456; that matches the documented behavior and is not a bug.
- Anything in the task text or documentation that was unclear: no; the README's note on which alias gets inferred as output, and on shard-budget accounting (data only, oversized tensors alone in their own shard), matched observed behavior exactly.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~15 minutes, most of it reading `docpack/README.md` and `help.txt` for the `assert` expressions and confirming `cast_`'s dtype semantics against the checkpoint before running.
