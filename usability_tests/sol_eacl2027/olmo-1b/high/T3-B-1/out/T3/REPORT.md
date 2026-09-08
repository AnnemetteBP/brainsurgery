# Participant self-report

- Final artifact path: `out/T3/`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): BrainSurgery tensor regexes use full-match semantics, so the projection selector was anchored by its complete structure and limited explicitly to layers 0 through 15.
- Pitfalls or surprises you hit (one line each): The two 412 MB float32 tensors correctly exceed the 256 MiB budget and are therefore stored alone; shard file sizes also include safetensors headers while the limit applies only to tensor data.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): Not applicable (condition B).
- Approximate time spent, if you can tell: About 5 minutes.
