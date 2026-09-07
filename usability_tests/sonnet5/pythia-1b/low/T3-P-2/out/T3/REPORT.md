## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The bf16 pattern must match exactly the 4 named projection matrices per layer and not incidentally catch biases (`.weight` suffix is required, not just a name substring).
  - Embeddings (206 MB each) exceed the 256 MiB shard budget and must each get a dedicated shard rather than triggering an error.
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes
