# T2 participant self-report

## Tools used and why

Tried `transformers`' `prune_heads` API first, since it's the route the
condition brief calls out for this task. It doesn't exist in the pinned
version (`transformers==5.12.1`): `PreTrainedModel.prune_heads` and
`find_pruneable_heads_and_indices` have been removed upstream, so neither
`GPT2Model` nor `GPT2LMHeadModel` expose it. `mergekit` (whole-tensor/layer
operations) and `torch-state-bridge` (key renaming) don't offer sub-tensor
column/row slicing either, so none of the condition's higher-level tools
actually fit this task as pinned.

Fell back to a plain script on `safetensors` + `torch`: load the state dict,
slice the three head-bearing tensors per layer using the exact block
boundaries given in TASK.md (5 kept heads then 6 kept heads within each
768-wide q/k/v segment, and analogous row blocks for `c_proj.weight`), leave
everything else untouched, assert the required shapes and the 160-tensor
count, then write with `safetensors.torch.save_file`.

## Verification

Independently re-derived the same column/row index lists in a throwaway
Python snippet against the raw input file and confirmed the output tensors
are bit-exact (`torch.equal`) for `h.0.attn.c_attn.weight/bias` and
`h.0.attn.c_proj.weight`, that the key set is unchanged, and that untouched
tensors (`c_proj.bias`, `attn.bias`) are identical to the input.

## Pitfalls / notes for the experimenter

- The suggested `prune_heads` route in `F-allowed.md` is stale for the pinned
  `transformers` version; discovering that cost one failed attempt
  (`AttributeError: 'GPT2LMHeadModel' object has no attribute 'prune_heads'`)
  before falling back to manual slicing.
- Conv1D `[in, out]` layout confirmed: `c_attn.weight` heads are column
  blocks (sliced on dim 1), `c_proj.weight` heads are row blocks (sliced on
  dim 0) — matches TASK.md, no surprises there once the block boundaries were
  taken as given rather than recomputed from head/dim arithmetic by hand.

## Required-checks enforcement

`out/T2/solution.py` asserts, before writing: the three per-layer shapes
(`c_attn.weight` → `[768, 2112]`, `c_attn.bias` → `[2112]`, `c_proj.weight` →
`[704, 768]`) for every layer (not just layer 0), the input has 160 tensors,
and the output has 160 tensors. Any failure raises `AssertionError` and
nothing is written.

## Attempts

1. `transformers` `GPT2Model.prune_heads(...)` — failed, `AttributeError`
   (method removed in `transformers==5.12.1`).
2. `transformers` `model.prune_heads(...)` on the top-level `GPT2LMHeadModel`
   — same failure, for the same reason.
3. `python3 out/T2/solution.py` (manual slicing script) — succeeded, wrote
   `out/T2/model.safetensors` with 160 tensors, checks passed.
