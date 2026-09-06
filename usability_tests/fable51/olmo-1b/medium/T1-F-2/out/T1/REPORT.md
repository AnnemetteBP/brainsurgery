# T1 self-report (condition F, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Input is sharded; loaded both shards via `model.safetensors.index.json` and merged into one dict before renaming.
  - Renumbering collisions were avoided by building a fresh output dict from a single old->new map instead of renaming in place.
- Anything in the task text or documentation that was unclear: the "Required checks" list says blocks 12..15 must not remain; the script additionally checks that the surviving indices are exactly 0..11 with 7 tensors each, and that each survivor is the same tensor (shape, dtype, storage) as its source.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file` / `save_file` for the shards and the single-file output.
  - `torch` 2.14.0: tensor objects only.
  - Plain Python `re` for block-index capture. mergekit and torch-state-bridge were considered but a 60-line script was simpler to make fail-loud and to verify bit-exactness with.
- Approximate time spent, if you can tell: about 2 minutes.
