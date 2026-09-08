## Participant self-report

- Final artifact path: `out/T1/model.safetensors`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each): Execution 1 failed before output because `assert count: 0` rejected a zero-match reference during reference resolution; replaced it with `not: { exists: ... }`.
- Pitfalls or surprises you hit (one line each): The `count` assertion cannot express an expected zero match because reference resolution itself requires at least one match.
- Anything in the task text or documentation that was unclear: Nothing in the task text was unclear; the zero-match behavior of `count` was not apparent from the assertion summary.
- Tools used (condition F): N/A (condition B).
- Approximate time spent, if you can tell: About 3 minutes.
