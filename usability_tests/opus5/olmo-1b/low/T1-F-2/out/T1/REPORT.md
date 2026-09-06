# T1 self-report

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: avoided entirely by building a fresh output dict
    keyed by the new names and failing on any duplicate key, rather than renaming
    in place in a shared dict.
  - Input is sharded, so keys must be resolved through `model.safetensors.index.json`;
    the output is required to be a single unsharded file.
  - `lm_head.weight` and `model.embed_tokens.weight` are stored as separate tensors
    in the index, so no shared-storage problem when saving with safetensors; I still
    call `.contiguous()` on every tensor before saving.
- Anything in the task text or documentation that was unclear: nothing material.
  The task fixes the drop set and the full remap explicitly, so there was no ambiguity.
- Tools used (condition F): `safetensors` 0.5.3 (`safe_open` for shard-wise lazy read,
  `save_file` for the single-file write) and `torch` 2.14.0 (tensor equality for the
  value-identity check). I did not use mergekit: its passthrough layer slicing works on
  contiguous ranges and would need four slices plus a model-level export, which adds a
  config round-trip and re-serialization risk for what is a pure key rename; a ~60-line
  script gives bit-exact copies and lets me enforce the required checks before writing.
  torch-state-bridge would fit the rename but adds a dependency for one regex.
- Approximate time spent, if you can tell: ~3 minutes.
