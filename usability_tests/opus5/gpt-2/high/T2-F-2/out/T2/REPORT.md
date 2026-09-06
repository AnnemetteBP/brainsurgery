# T2 participant self-report (condition F, GPT-2 124M)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - `F-allowed.md` names `transformers.prune_heads` as the plausible route for T2, but transformers 5.12.1 has removed it: `GPT2LMHeadModel.prune_heads` does not exist, and `transformers.pytorch_utils` no longer exports `prune_conv1d_layer` or `find_pruneable_heads_and_indices` (only `prune_linear_layer`, which is the wrong layout for Conv1D). The suggested route is not available at the pinned version.
  - Even if it existed, that route could not satisfy the task: transformers 5.x GPT-2 has dropped the `attn.bias` causal-mask buffer, so a `save_pretrained` round-trip yields 148 tensors, not the required 160, and prefixes every key with `transformer.`.
  - `pruned_heads` in `config.json` no longer resizes modules before loading in 5.12.1 — `from_pretrained` reports 36 shape MISMATCHes and raises. So there is no in-library way to *load* the pruned checkpoint as an 11-head GPT-2 either; setting `n_head=11` instead fails validation because GPT-2 derives `head_dim = embed_dim / n_head` and 768 is not divisible by 11.
  - Conv1D `[in, out]`: the same head 5 lives in *columns* of `c_attn.weight` (three times, once per q/k/v segment) but in *rows* of `c_proj.weight`. Slicing the wrong axis on `c_proj` would still produce a `[704, 768]`-looking result only if you also transposed, so shape checks alone do not catch this — I checked it semantically instead.
  - A float32 equivalence check between head-5-masked base attention and the pruned weights shows 2e-5 max abs difference, which initially looks like a slicing bug; it is just accumulation order over a 768- vs 704-wide reduction (relative ~3e-7). Redone in float64 the difference is exactly 0.
- **Anything unclear in the task text or documentation:** nothing material. The task gives the keep-ranges explicitly, which removes the ambiguity about whether q/k/v are fused per-head or per-projection. "Loadable as the same architecture with 11 heads per layer" is not literally achievable with the pinned transformers (see above); I read it as a statement about tensor layout, which the checkpoint satisfies.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load and save the checkpoint. Direct, preserves the exact key set and dtypes, and `save_file` is the only thing that can emit exactly the 160 input tensors.
  - `torch` 2.14.0 — `index_select` for the gather, and `torch.equal` for the bit-exact untouched-tensor check.
  - `transformers` 5.12.1 — only investigated; its pruning API is gone at this version (see pitfalls). Not used in the final artifact.
  - Not used: `mergekit` (merging/layer-slicing, no intra-tensor axis surgery), `peft` (adapters), `torch-state-bridge` (key rewriting, not value slicing). None of them can slice head blocks out of a fused projection.
- **Approximate time spent:** ~10 minutes, most of it on verification rather than the edit.

## What the script enforces before writing

Derivation and spec are cross-checked against each other: the keep-indices are computed from head geometry (`n_head=12`, `head_dim=64`, prune head 5) and asserted equal to the literal ranges from TASK.md. Then, before `save_file`:

- input shape of every head-bearing tensor is verified as `[768, 2304]` / `[2304]` / `[768, 768]`;
- all 36 expected head-bearing keys must be present in the input;
- the four required checks: `h.0.attn.c_attn.weight` is `[768, 2112]`, `h.0.attn.c_attn.bias` is `[2112]`, `h.0.attn.c_proj.weight` is `[704, 768]`, and exactly 160 tensors;
- the same three shape checks for all 12 layers, not just layer 0;
- every non-head-bearing tensor is bit-identical to the input, and no tensor changes dtype.

Any failure raises before anything is written, so no partial output is possible.

## Post-hoc verification (separate from the script)

- Key set identical to the input; all 160 dtypes unchanged; 36 tensors resliced, 124 bit-identical; every resliced tensor bit-equal to the independently recomputed spec slice.
- Semantic check on layer 0: attention computed from the pruned weights with 11 heads equals attention from the base weights with head 5's output zeroed — difference exactly 0.0 in float64.
