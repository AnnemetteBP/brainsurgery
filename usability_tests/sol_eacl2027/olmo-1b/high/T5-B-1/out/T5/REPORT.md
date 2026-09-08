## Participant self-report

- Final artifact path: `out/T5/`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each): None exited with an error; execution 1 was superseded after the post-run index audit showed that a literal 512 MiB packing cap placed `model.embed_tokens.weight` with one additional tensor instead of alone.
- Pitfalls or surprises you hit (one line each): The stated 412 MB embedding tensor is smaller than 512 MiB, so `shard: 512MB` does not force it into a standalone shard; using the exact 393 MiB tensor-data size does.
- Anything in the task text or documentation that was unclear: The claim that the 412 MB embedding tensors are larger than the 512 MiB cap is numerically inconsistent, although the required standalone layout is clear.
- Tools used (condition F): N/A (condition B; BrainSurgery plan only).
- Approximate time spent, if you can tell: About 7 minutes.
