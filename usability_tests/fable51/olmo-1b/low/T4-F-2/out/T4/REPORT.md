# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The base is sharded while the fine-tunes are single files, so the loader reads the index's `weight_map` and checks for duplicate keys across shards.
  - Nothing else; the MLP regex is anchored so it cannot overmatch, and every check aborts before any arithmetic or writing.
- Anything in the task text or documentation that was unclear: nothing.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: tensor arithmetic in float32 and bit-exact `torch.equal` comparison of the 66 shared tensors.
  - `safetensors` 0.5.3: reading the shards and single-file checkpoints, writing the single output file.
  - Not used: `mergekit` task arithmetic. It would have produced the merge but not the mandatory three-way pre-verification of the non-MLP tensors, and it writes HF-style sharded directories rather than one guaranteed file, so a script was shorter and fully checkable.
- Approximate time spent, if you can tell: about 2 minutes.
