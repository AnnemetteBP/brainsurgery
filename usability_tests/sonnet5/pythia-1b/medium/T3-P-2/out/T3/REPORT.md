## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - After upcasting `gpt_neox.embed_in.weight`/`embed_out.weight` from float16
    to float32 they grow to ~412 MB each (double the 206 MB stated for the
    float16 input), well over the 256 MiB shard limit; per spec they still go
    alone in their own shard rather than being split further.
- Anything in the task text or documentation that was unclear: none — the
  "206 MB each" figure refers to the float16 input size, worth noting since
  the output tensors end up larger after the float32 upcast.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~5 minutes
