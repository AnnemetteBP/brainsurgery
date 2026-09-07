# T2 run record — Participant self-report

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none — the single
  execution succeeded and passed all checks.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious library route, `transformers` `prune_heads`, does not exist for
    this architecture: `GPTNeoXAttention` in transformers 5.12.1 has no
    `prune_heads` and the module never references `find_pruneable_heads_and_indices`,
    so the `F-allowed.md` hint for T2 is a dead end here.
  - Even if it existed, `prune_heads` could not express the result: GPT-NeoX
    derives `head_size = hidden_size // num_attention_heads`, and 2048/7 is not
    an integer, so a 7-head Pythia-1B is not representable by the stock config.
    That also means the target must be produced by direct tensor surgery.
  - The real trap is the fused `query_key_value` layout: GPT-NeoX interleaves
    per head (`[q_h | k_h | v_h]` blocks of 768 rows), not `[Q | K | V]` segments.
    Treating it as three 2048-row segments and dropping rows 1280..1535 of each
    would give the right *shape* `[5376, 2048]` and load fine while producing
    garbage attention — a shape check alone does not catch this.
  - `index_select` returns a non-contiguous view for the column slice of
    `dense.weight`; `.contiguous()` before `save_file` avoids a safetensors
    save error.
  - float16 had to survive the round trip untouched — I loaded with
    `safetensors.torch.load_file` and never cast, so dtypes are preserved
    verbatim (including the `uint8` `attention.bias` buffer and the 0-d
    `masked_bias`).

- **Anything in the task text or documentation that was unclear:** nothing
  material. The task text was unusually explicit about the interleaved layout
  and the exact row/column ranges to keep, which removed the main ambiguity.
  Minor: "loadable as the same architecture with 7 heads per layer" is not
  literally achievable with the stock `GPTNeoXConfig` (see above), and the task
  asks for only `model.safetensors`, so I wrote no `config.json`.

- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — `load_file` / `save_file`; the only thing needed to
    read and write the checkpoint while preserving dtypes exactly.
  - `torch` 2.14.0 — `index_select` / `arange` / `cat` for the slicing and for
    the bit-exact verification pass.
  - `transformers` 5.12.1 — only *inspected* (to establish that `prune_heads`
    is unavailable for GPT-NeoX); not used to produce the output.
  - Not used: `mergekit` (layer-granular merges/slices, cannot address tensor
    sub-blocks), `peft` (adapters, unrelated), `torch-state-bridge` (renames
    keys, does not reshape values). None of them can slice *within* a tensor,
    which is the entire task, so a ~100-line script on top of
    safetensors+torch was both the smallest and the safest route.

- **Approximate time spent, if you can tell:** ~5 minutes.

## What the script enforces before writing

- The computed keep-indices are asserted equal to the literal ranges from the
  task (`0..3839` + `4608..6143` for qkv rows, `0..1279` + `1536..2047` for
  dense columns), so a wrong block boundary fails at the index level rather
  than silently producing a correctly-shaped, wrong-valued checkpoint.
- Input shapes of all three head-bearing tensors are checked per layer.
- The required output-shape checks (`[5376, 2048]`, `[5376]`, `[2048, 1792]`)
  are enforced on layer 0 *and* on all 16 layers.
- Tensor count is asserted to be exactly 244 and to be unchanged from the input.
- `attention.dense.bias` is asserted to still be `[2048]` (untouched).

All of these raise `SystemExit` before `save_file` is reached.

## Independent post-hoc verification (separate from the script)

Reloaded both checkpoints and confirmed: identical key sets; 48 pruned tensors
bit-exact against independently recomputed `torch.cat` slices; the other 196
tensors byte-identical to the input; all dtypes preserved.
