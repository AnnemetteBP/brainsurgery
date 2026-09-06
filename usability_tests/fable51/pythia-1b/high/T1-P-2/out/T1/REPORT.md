# Participant self-report: T1 (Pythia-1B depth pruning), condition P

- Final artifact path: `out/T1/solution.py` (output checkpoint `out/T1/model.safetensors`, 184 tensors, 12 blocks)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collisions were avoided by building a fresh destination dict from an explicit old->new map and failing on any duplicate destination key, rather than renaming in place.
  - The layer regex is anchored (`^gpt_neox\.layers\.(\d+)\.`) so `gpt_neox.layers.1.` cannot match blocks 10..15, and the three buffers (`attention.bias`, `masked_bias`, `rotary_emb.inv_freq`) are carried along with the parameters because matching is by block prefix, not by tensor kind.
  - All checks run on the in-memory result before writing; the file is written to a temp path and renamed, so a failed check leaves no output.
- Anything in the task text or documentation that was unclear:
  - "No tensor of blocks 12, 13, 14, 15 remains" refers to output-namespace indices (i.e. nothing >= 12 after renumbering), not the removed old blocks 2, 6, 10, 14; I checked both (indices are exactly 0..11 and each has 15 tensors).
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes.
