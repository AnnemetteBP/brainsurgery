# Participant self-report: T1 (Pythia-1B depth pruning), condition P

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Avoided the renumbering-collision hazard by building a fresh output dict from an old->new index map instead of renaming in place.
  - Anchored the layer regex with `^...\.(\d+)\.` so `layers.1.` cannot match `layers.10.`–`layers.15.`.
- Anything in the task text or documentation that was unclear: the "Required checks" say no tensor of blocks 12..15 may remain, which is implied by the contiguity check; I implemented both explicitly. The qkv-layout description is not needed for this task.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
