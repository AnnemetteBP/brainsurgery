# T1 self-report (condition F, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/solution.py` (produces `out/T1/model.safetensors`, 86 tensors, blocks 0..11)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The input is sharded across two files, so keys were gathered from both shards via the index's `weight_map` and checked for cross-shard duplicates.
  - Renumbering was done by building a fresh dict keyed by the new names (old->new map computed once from the ordered survivor list) instead of renaming in place, which removes the collision hazard entirely; a collision check on the destination dict is still enforced.
  - The output is staged to a `.tmp` path, re-read from disk, re-verified, and only then renamed into place, so a failing check leaves no output file.
- Anything in the task text or documentation that was unclear:
  - "Required checks" says no tensor of blocks 12..15 may remain, which is a weaker condition than the required result (surviving indices must be exactly 0..11); the script enforces both, plus bit-exact equality of every output tensor against its source tensor.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 (`safe_open`, `torch.save_file`): direct shard reading and single-file writing; the task is a pure key rewrite, so a plain script over the safetensors API is the shortest path with no intermediate model instantiation or config edits.
  - `torch` 2.14.0: tensor equality for the integrity checks.
  - Considered and rejected: `mergekit` passthrough slicing (needs a config with a modified `num_hidden_layers` and writes sharded output plus config files, then would need a re-pack step to a single file), `torch-state-bridge` (regex renaming would work but adds nothing over a 12-entry index map), `transformers` (no need to instantiate the model).
- Approximate time spent, if you can tell: about 2 minutes, dominated by loading and writing ~4 GB of float32 weights.
