# T1 self-report

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why: none.
- Pitfalls or surprises you hit:
  - Renumbering collision hazard: avoided by building a fresh output dict keyed by
    the new names instead of renaming in place, plus an explicit collision check.
  - `attn.bias` is a causal-mask buffer, not a weight, but it is one of the 13
    per-block tensors, so it must be dropped/renumbered with its block.
  - Only the block index may change; I split on the first dot after `h.` and kept
    the remainder verbatim rather than regex-substituting inside the whole name.
- Anything unclear: nothing; the required key set and count were fully specified.
- Tools used (condition F): `safetensors` 0.5.3 (load/save), `torch` 2.14.0 (tensor
  values via `safetensors.torch`), Python stdlib `re`. I skipped mergekit: its
  passthrough layer slicing renumbers via a HuggingFace model export, which would
  rewrite the checkpoint (tied/renamed keys, no `attn.bias` buffer) and cannot
  guarantee the exact 121-key, bit-exact result the grader asks for. A direct
  key rewrite is smaller, exact, and lets the required checks gate the write.
- Approximate time spent: ~3 minutes.
