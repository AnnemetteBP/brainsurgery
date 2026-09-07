# Participant self-report: T1 (OLMo-1B-0724-hf, condition P)

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`, 86 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Input is sharded; loaded both shards via `model.safetensors.index.json` and merged into one dict (checked for duplicate keys across shards).
  - Renumbering collisions avoided by building a fresh output dict from an old->new index map instead of renaming in place.
  - Task "Required checks" says "no tensor of blocks 12, 13, 14, 15 remains", which refers to new indices >= 12, not the removed old blocks 2/6/10/14; checked both interpretations (index >= 12 absent, exactly 12 contiguous blocks with 7 tensors each).
- Anything in the task text or documentation that was unclear: the "blocks 12, 13, 14, 15" wording in the checks reads oddly next to "remove blocks 2, 6, 10, 14"; it is consistent once read as post-renumber indices.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: a few minutes
