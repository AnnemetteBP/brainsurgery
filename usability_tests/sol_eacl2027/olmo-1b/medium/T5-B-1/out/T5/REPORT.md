## Participant self-report

- Final artifact path: `out/T5/`
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - Execution 1: the `matmul` paired-source rewrite treated escaped punctuation as literal text, so the rewritten LoRA A tensor name did not exist.
  - Execution 2: the merge completed, but `assert count: 0` failed because `count.of` rejects an empty match set before comparing its count; changed this to `not: { exists: ... }`.
- Pitfalls or surprises you hit (one line each):
  - Rewrite fields synthesize names rather than acting as regexes, so punctuation needed in a rewritten source name must either be literal and unescaped or carried through captures.
  - Absence must be checked with `not` plus `exists`, rather than `count` equal to zero.
- Anything in the task text or documentation that was unclear: The general capture-based ternary rewrite behavior was documented, but it was not explicit that regex escapes in non-driving rewrite fields are emitted literally.
- Tools used (condition F): N/A (condition B).
- Approximate time spent, if you can tell: About 5 minutes.
