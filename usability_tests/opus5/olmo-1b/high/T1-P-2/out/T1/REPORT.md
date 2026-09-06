# T1 participant self-report (condition P)

- **Final artifact path:** `out/T1/model.safetensors` (86 tensors), produced by
  `out/T1/solution.py`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why (one line each):** none; the first
  execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The renumbering collision hazard is the whole task: renaming in place (e.g.
    `state[new] = state.pop(old)` over the blocks) would let block 3 land on the
    surviving block 2 depending on iteration order, so I built a fresh output
    dict keyed by the new names and made a duplicate new name a hard failure
    rather than a silent overwrite.
  - The input is sharded, so the state dict has to be stitched from both shard
    files via `model.safetensors.index.json`; I checked the index and the shard
    contents agree in both directions instead of trusting either alone.
  - `model.layers.<i>.` needs escaped dots and an anchored match; an unescaped
    or unanchored pattern would also be satisfied by unintended text, and
    `model.layers.1` as a prefix would otherwise catch layers 10..15.
  - `safetensors.save_file` rejects non-contiguous tensors, so surviving
    tensors are made contiguous before saving (none actually needed it here).
  - Output is a single file while the input is sharded, so no index file is
    written alongside it, per the task's "single file" wording.
- **Anything in the task text or documentation that was unclear:** the required
  check "no tensor of blocks 12, 13, 14, 15 remains" is trivially implied by
  "indices run 0..11", so I read it as a check on the output key set (no block
  index >= 12 survives) and implemented it that way. The task also does not say
  whether to copy the HF config/tokenizer files or emit a 12-layer
  `config.json`; it names exactly one output file, so I wrote only that.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors
  0.5.3 only).
- **Approximate time spent, if you can tell:** roughly 5 minutes — one pass
  reading the task and the shard index, one pass writing the script, one run.
