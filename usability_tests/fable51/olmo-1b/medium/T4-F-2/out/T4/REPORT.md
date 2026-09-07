# T4 participant self-report (condition F)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base is sharded (two files plus an index) while the fine-tunes are single files, so the loader reads the index's `weight_map` for the base and a single file otherwise.
  - The output file needs `metadata={"format": "pt"}` to be loadable by `transformers`; added it to `save_file`.
- Anything in the task text or documentation that was unclear: nothing material. The spec is explicit about lambda, the tensor set, and the checks.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `safe_open` for reading all three checkpoints and `save_file` for the single-file output; gives full control over which tensors are touched.
  - `torch` 2.14.0: float32 arithmetic for the task-vector merge and `torch.equal` for the bit-exact shared-tensor verification.
  - Not used: `mergekit` task arithmetic. It does not perform the required precondition check (that all non-MLP tensors are identical across the three checkpoints) and would apply the merge to every tensor rather than only the 48 MLP tensors, so it could not satisfy step 1 or guarantee bit-exact copies of the 66 unchanged tensors without extra scripting anyway.
- Approximate time spent, if you can tell: about 3 minutes wall clock including the 14 s run.

## Checks enforced by the script

- Identical key sets across base, ft1, ft2, and exactly 114 tensors.
- Exactly 48 tensors match the MLP pattern (`model.layers.<i>.mlp.{gate,up,down}_proj.weight`).
- Shape and dtype equality for every tensor across the three checkpoints.
- Bit-exact equality (`torch.equal`) of every non-MLP tensor across the three checkpoints; the script aborts before merging if any differs.
- Merge counter equals 48; output dict size equals 114.
- After writing, the file is re-opened and the key set, shapes, dtypes, and bit-exactness of the 66 unchanged tensors are re-verified against the base.

Each task vector is computed against the untouched in-memory base tensor, so ordering of the two additions cannot contaminate the result.
