# Participant self-report

- Final artifact path: out/T3/solution.py
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Had to distinguish `h.<i>.attn.bias` (non-parameter causal-mask buffer, dropped) from the actual projection weights via separate, non-overlapping regexes rather than one broad `.*weight` pattern.
  - Sharding required greedy bin-packing by tensor byte size (numel * element_size) with the oversized `wte.weight` (154 MB) placed alone in its own shard since it exceeds the 64 MiB cap.
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes
