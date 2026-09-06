## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `matmul`/`add_`/`permute` support regex `from`/`to` with capture-group
    rewrites across all 12 layers in one transform, but the rewrite strings
    are plain substitution text, not nested regexes: writing `\.` in a
    replacement (e.g. `h.\1.attn...`) leaves a literal backslash in the name
    and the lookup for the paired tensor fails with "source_b missing". Fixed
    by using literal dots in every `to`/`from_b` rewrite string.
  - Conv1D layout `[in, out]` vs. the adapter's `nn.Linear` convention: had to
    transpose `B @ A` (shape `[out, in]`) before adding, matching
    `fan_in_fan_out = true`.
- Anything in the task text or documentation that was unclear: none; the
  README's note on regex rewrites for `add`/`copy` made it clear the same
  mechanism applies to `matmul`/`permute`/`add_`, though the escaping
  behavior of the replacement string had to be discovered by trial.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes
