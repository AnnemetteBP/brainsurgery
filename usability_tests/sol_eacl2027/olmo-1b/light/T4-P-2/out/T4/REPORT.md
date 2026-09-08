# Participant self-report

- Final artifact path: `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The base checkpoint is sharded while both fine-tunes are single files, so tensor access had to follow the base index and avoid assuming identical file layout.
- Pitfalls or surprises you hit (one line each): Each task vector was computed against the original base tensor in one expression to avoid order-dependent sequential merging.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): Not applicable (condition P).
- Approximate time spent, if you can tell: About 5 minutes.
