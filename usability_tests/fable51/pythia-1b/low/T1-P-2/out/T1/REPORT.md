# Participant self-report: T1 (Pythia-1B depth pruning), condition P

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering into a fresh dict (old index -> new index map) avoids the in-place shift collision hazard entirely; an explicit collision check was added anyway.
  - Regex anchored on `gpt_neox.layers.<i>.` with escaped dots so `layers.1.` cannot match `layers.10.`.
- Anything in the task text or documentation that was unclear: the "Required checks" section says "no tensor of blocks 12, 13, 14, 15 remains", which is phrased in terms of new indices; I interpreted it as "no surviving index >= 12" and additionally checked indices are exactly 0..11 with 15 tensors each.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes
