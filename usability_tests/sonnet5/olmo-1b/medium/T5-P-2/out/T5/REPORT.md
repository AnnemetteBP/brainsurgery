## Participant self-report

- Final artifact path: out/T5/solution.py (output written to out/T5/)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, it succeeded on the first run
- Pitfalls or surprises you hit (one line each):
  - The 512 MiB shard limit is on tensor data only, not file size; the safetensors header pushes the on-disk file a few hundred bytes past 536,870,912 even when the packed tensor bytes exactly hit the limit, so I verified against summed tensor bytes rather than `os.path.getsize`.
- Anything in the task text or documentation that was unclear: none, the mapping from adapter names to base names and the scale formula were spelled out explicitly.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes
