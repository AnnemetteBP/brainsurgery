# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first execution succeeded
- Pitfalls or surprises you hit (one line each):
  - The two ~412 MB embedding/lm_head tensors each exceed the 256 MiB shard
    budget on their own, so they need their own shard rather than being
    bin-packed with anything else.
  - Had to be careful that the regex for projection matrices only matches
    the 7 named per-layer weight names and not, e.g., embeddings or a
    hypothetical norm/bias tensor.
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes
