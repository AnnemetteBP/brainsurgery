# T1 self-report (Pythia-1B, condition F)

- Final artifact path: `out/T1/solution.py` (produces `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None. Renaming into a fresh dict (rather than renaming in place) removes the ordering/collision hazard entirely; a collision check still guards it.
  - Buffers (`attention.bias`, `masked_bias`, `rotary_emb.inv_freq`) are matched by the same `gpt_neox.layers.<i>.` prefix, so they move with their block without special handling.
- Anything in the task text or documentation that was unclear:
  - The required check "no tensor of blocks 12, 13, 14, 15 remains" refers to post-renumbering indices; I read it that way (any surviving index >= 12 fails).
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 (`safe_open`, `save_file`): direct key-level read/write with metadata preserved; no model instantiation needed for a pure rename.
  - `torch` 2.14.0: tensor container and `torch.equal` for the post-hoc bit-exact check.
  - Not used: mergekit (passthrough slicing would rewrite via HF save and re-derive buffers/config; more moving parts than a 60-line script), torch-state-bridge (rename rules are simpler as an explicit dict here), transformers (loading the model would drop the mask buffers from the checkpoint).
- Approximate time spent, if you can tell: ~2 minutes
