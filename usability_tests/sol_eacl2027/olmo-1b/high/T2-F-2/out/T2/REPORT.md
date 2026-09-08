# Participant self-report

- Final artifact path: `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None; the first artifact-producing execution succeeded.
- Pitfalls or surprises you hit (one line each): The Q/K/V head blocks are on the row axis, while the O projection head blocks are on the column axis.
- Pitfalls or surprises you hit (one line each): The 4.8 GB input is sharded, so the solution loads each shard once and retains memory-mapped unmodified tensors while allocating only the pruned projections.
- Anything in the task text or documentation that was unclear: Nothing.
- Tools used (condition F): `torch` 2.14.0, for exact tensor slicing/concatenation; `safetensors` 0.5.3, for memory-mapped shard loading and single-file serialization; Python standard library, for reading the HuggingFace shard index and strict key matching.
- Approximate time spent, if you can tell: About 4 minutes.

An independent verification pass (`out/T2/verify.py`) compared all 114 saved
tensors with their expected source or reconstructed pruned value and confirmed
the exact key set, shapes, dtypes, and bit-exact values.
