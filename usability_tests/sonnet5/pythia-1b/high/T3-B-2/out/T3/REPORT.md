## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `cast_`/`delete` targets must not be sliced and require full-match regex on the dot-separated tensor name, so the "everything else" cast needed a negative lookahead (`(?!proj-pattern$).*`) rather than a positive list, to avoid re-deriving the projection pattern twice and risking drift between the two casts.
  - There is no assert operator that counts tensors by dtype directly (`count`/`dtype` both take a name-based `of` reference); "exactly 64 bfloat16 tensors" had to be proven indirectly by asserting the projection-name pattern matches exactly 64 tensors, that all of them are bfloat16, and that the complementary (exhaustive, disjoint) pattern is entirely float32 — the buffer deletion had to happen first so the two casts partition the checkpoint cleanly.
  - Deleting the 48 buffers before casting avoids having to exclude them from the "everything else" float32 pattern (they're `uint8`/rotary-frequency buffers that would otherwise need their own carve-out).
- Anything in the task text or documentation that was unclear: none; the README's notes on regex full-matching and shard budgeting (single oversized tensor gets its own shard) matched observed behavior exactly.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: about 15 minutes, most of it reading `docpack/help.txt` for `cast_`/`delete`/`assert` semantics before writing the plan.
