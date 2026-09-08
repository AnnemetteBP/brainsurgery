# Participant self-report

- Final artifact path: `out/T3/`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): Filesystem sizes include safetensors headers, while the 256 MiB limit applies only to tensor data, so shard compliance was checked from tensor shapes and element sizes rather than file sizes.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): PyTorch 2.14.0 for the required float32-to-bfloat16 conversion and tensor validation; safetensors 0.5.3 for streaming input reads and sharded checkpoint writes; Python standard-library JSON for the HuggingFace index.
- Approximate time spent, if you can tell: About 5 minutes.
