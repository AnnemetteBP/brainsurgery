# T2 self-report (condition F, GPT-2 124M, structured head pruning)

- **Final artifact path:** `out/T2/solution.py` (output `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - Conv1D layout: `c_attn.weight` is `[in, out]`, so heads are *columns* there but *rows* in `c_proj.weight` — the two head-bearing tensors are sliced on opposite axes.
  - `c_attn` is fused q|k|v, so head 5 has to be dropped three times, once per 768-wide segment, at offsets 320..383, 1088..1151 and 1856..1919.
  - Suffix matching on `.attn.c_attn.*` / `.attn.c_proj.weight` must not catch `mlp.c_proj.weight` (also 768-wide on one axis) or `attn.c_proj.bias`/`attn.bias`; I asserted `mlp.c_proj.weight` stayed `[3072, 768]` and the mask buffer stayed `[1, 1, 1024, 1024]`.
  - `index_select` returns fresh storage, but I still `.contiguous().clone()` the edited tensors so safetensors never sees a view into the source buffer.
  - Checkpoint is a bare `GPT2Model` state dict (no `transformer.` prefix, no `lm_head`), so no tied-weight sharing issue on save.
- **Anything in the task text or documentation that was unclear:** nothing; the explicit kept-column ranges made the layout unambiguous, and I used them as an independent cross-check rather than as the implementation.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save the checkpoint; it is the file format, so unavoidable.
  - `torch` 2.14.0 — `index_select` for the head-block gather and `torch.equal` for verification.
  - Considered and rejected: `transformers` 5.12.1 `prune_heads`. It is the advertised route, but it works on an instantiated `GPT2Model`, rewrites `Conv1D` modules and records `pruned_heads` in the config; re-exporting would risk dropping/renaming buffers (`attn.bias`) and changing the key set, while grading demands exactly the original 160 keys and bit-exact values. A 90-line slicing script gives direct control over axis, block order and the checks. `mergekit` and `peft` do not express intra-tensor head slicing; `torch-state-bridge` rewrites keys, not values.
- **Approximate time spent:** ~5 minutes.

## Checks enforced by the run (all before writing)

Per-layer shape asserts on all 36 edited tensors plus the untouched `attn.c_proj.bias`, `attn.bias` and `mlp.c_proj.weight`; the four required checks (`h.0.attn.c_attn.weight == [768, 2112]`, `h.0.attn.c_attn.bias == [2112]`, `h.0.attn.c_proj.weight == [704, 768]`, 160 tensors); key set and dtype identical to the input; boundary spot-checks that kept column 320 is source column 384 and kept row 320 of `c_proj` is source row 384. After writing, the file is reloaded and the four required checks are re-asserted. Any failure raises `AssertionError` and no file is written.

Separately (not part of the artifact) I rebuilt every edited tensor by `torch.cat` of the literal ranges given in TASK.md and confirmed bit-exact equality with the written file, and that all other 124 tensors are bit-identical to the input.
