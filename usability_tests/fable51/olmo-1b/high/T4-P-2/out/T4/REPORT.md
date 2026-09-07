# T4 participant self-report (condition P)

- Final artifact path: `out/T4/solution.py` (output checkpoint `out/T4/model.safetensors`, 114 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 succeeded.
- Pitfalls or surprises you hit (one line each):
  - Base is sharded (two files plus index) while ft1/ft2 are single files, so the script needed a small reader that resolves names via the index and cross-checks the index against the shards.
  - Three float32 checkpoints are about 15 GB total; the script streams tensors with `safe_open` and only holds the output state dict in memory, rather than loading all three at once.
  - The shared-tensor check is done before any merge arithmetic, and each task vector is computed from the base tensor read from disk, so ordering cannot contaminate the second task vector.
- Anything in the task text or documentation that was unclear: nothing material. The task says "identical" for the shared check; I treated that as bit-exact (`torch.equal`) plus matching shape and dtype.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: about 3 minutes of authoring; the run took about 13 s wall clock.
