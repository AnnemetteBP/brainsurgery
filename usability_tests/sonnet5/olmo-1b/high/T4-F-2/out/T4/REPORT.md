# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Base checkpoint is sharded (`model.safetensors.index.json` + two shard files) while ft1/ft2 are single files, so loading had to go through the index rather than a single `load_file` call.
  - The non-MLP equality check needs to compare shape *and* dtype before `torch.equal`, since a shape mismatch would otherwise raise instead of failing the intended assertion cleanly.
- Anything in the task text or documentation that was unclear: none; the formula, tensor layout, and required checks were unambiguous.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain script. Considered `mergekit` task-arithmetic, but it doesn't have a direct out-of-the-box hook for the required pre-merge three-way equality verification (bit-exact match of all non-MLP tensors across base/ft1/ft2) or for asserting exact counts (48 merged, 114 total) before writing output; a ~130-line explicit script made every required check auditable and kept the per-tensor arithmetic (each task vector against the unmodified base, not sequential) obviously correct by construction.
- Approximate time spent, if you can tell: a few minutes of scripting plus verification.
