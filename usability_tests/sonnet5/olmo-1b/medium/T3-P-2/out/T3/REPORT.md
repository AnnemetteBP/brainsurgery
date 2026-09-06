# Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Had to be careful that the projection-matrix regex was anchored (`^...$`)
    so it couldn't accidentally match `lm_head.weight` or `embed_tokens.weight`,
    both of which end in `.weight` too.
  - Bin-packing the shards needed a special case: the two oversized tensors
    (embed_tokens, lm_head, 412 MB each) must go alone in their own shard even
    though that exceeds the 256 MiB budget, per the task's explicit exception;
    everything else uses first-fit-decreasing packing under the budget.
- Anything in the task text or documentation that was unclear: none; the task
  spelled out the exact tensor list, sizes, and shard exception clearly.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes
