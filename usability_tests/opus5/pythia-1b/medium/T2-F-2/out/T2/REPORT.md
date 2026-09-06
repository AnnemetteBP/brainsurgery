# T2 run record — Participant self-report

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded and wrote the output.
- **Pitfalls or surprises you hit (one line each):**
  - GPT-NeoX fuses q/k/v *interleaved per head* (768-row blocks of q|k|v), not as `[q | k | v]` segments; slicing head 5 as `rows 5*256..` of three separate segments would have been the natural wrong guess.
  - The two head-bearing axes point opposite ways: `query_key_value` heads are **rows** (dim 0, output side), `dense` heads are **columns** (dim 1, input side), because both use the `nn.Linear` `[out, in]` layout.
  - `torch.cat` of two slices already yields a contiguous tensor, so safetensors accepted the result with no `.contiguous()` surprises; I call it explicitly anyway.
  - The result is **not loadable by stock `transformers`**: `GPTNeoXAttention.__init__` hard-codes `nn.Linear(hidden_size, 3 * hidden_size)` and derives `head_size = hidden_size // num_attention_heads`, ignoring `config.head_dim`. A 7-head / 256-dim / 2048-hidden GPT-NeoX has no valid stock config, so "loadable as the same architecture with 7 heads" holds only for the tensor layout, not for an unpatched HF class.
  - Consequently `transformers.prune_heads` — the route `F-allowed.md` suggests for T2 — is unusable here: `GPTNeoXAttention` implements no `prune_heads`, and `PreTrainedModel.prune_heads` needs `find_pruneable_heads_and_indices` plus per-class `prune_linear_layer` support that GPT-NeoX's fused qkv does not provide.
- **Anything in the task text or documentation that was unclear:** nothing blocking — the spec pinned the interleaved layout, the exact kept ranges and the final shapes, which removed all layout guesswork. The only tension is the "loadable as the same architecture with 7 heads" claim versus what stock `transformers` can actually instantiate (above); I treated the explicit shape spec as authoritative.
- **Tools used (condition F):**
  - `torch` 2.14.0 — tensor slicing (`cat` of kept blocks) and the equality checks.
  - `safetensors` 0.5.3 — `load_file` / `save_file`, preserving float16 dtypes and the `{"format": "pt"}` metadata.
  - `transformers` 5.12.1 — **verification only**, not for the edit: I read `modeling_gpt_neox.py` to confirm the layout, and ran a functional equivalence check (below).
  - **Why a plain script rather than a toolkit:** the task is a deterministic, bit-exact slice of 48 named tensors with a copy-through of the other 196. `prune_heads` is unavailable for GPT-NeoX (above), mergekit operates on whole layers/merge arithmetic and has no intra-tensor slicing, and `torch-state-bridge` rewrites *keys*, not values. Every allowed toolkit would have needed a script around it anyway, with less control over dtype and bit-exactness.
- **Checks enforced before writing (the run aborts if any fails):** per-layer input shapes; per-layer output shapes `[5376, 2048]` / `[5376]` / `[2048, 1792]`; dtype unchanged on every edited tensor; kept blocks bit-equal to the corresponding source slices in order; exactly 48 tensors touched; key set unchanged; every untouched tensor identical; and the four literal required checks from TASK.md (layer-0 shapes and the 244-tensor count).
- **Post-write verification (separate from the script):** re-loaded both checkpoints and confirmed bit-exact equality against independently recomputed slices and byte-identity of all 196 untouched tensors; then a functional check — the pruned checkpoint's logits match the original model with head 5's `dense` columns zeroed (the mathematically equivalent ablation) to 9.1e-6 in float32, with identical argmax. A wrong block boundary would fail this by a wide margin.
- **Approximate time spent:** ~10 minutes, most of it on verification rather than the edit.
