# T2 self-report (condition F)

- **Final artifact path:** `out/T2/solution.py` -> `out/T2/model.safetensors`
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  - Execution 1: `failed_assertion` — my own blanket "every tensor is float16" sanity check fired on `gpt_neox.layers.0.attention.bias`, which is a `uint8` causal-mask buffer, not a weight.
- **Pitfalls or surprises you hit (one line each):**
  - The GPT-NeoX `query_key_value` layout is per-head interleaved (768-row blocks of q|k|v), not `[q | k | v]` segments — slicing it as three contiguous segments would silently produce a loadable but wrong checkpoint.
  - Non-float16 attention buffers exist in the checkpoint, so a global dtype assertion is wrong; the correct invariant is per-key dtype preservation.
  - Sliced tensors must be `.contiguous()` before `save_file`, otherwise safetensors can refuse or write unexpected strides.
- **Anything in the task text or documentation that was unclear:** Nothing significant. The task text spelled out the interleaved layout and the exact keep-ranges, which removed the main ambiguity; it did not mention that some untouched tensors are not float16, which is what tripped my extra check.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — direct `safe_open` / `save_file` for load and save. The task requires a bit-exact, key-for-key rewrite of one file, which is exactly this API's scope.
  - `torch` 2.14.0 — index-tensor slicing and `torch.equal` for verification.
  - Deliberately not used: `transformers.prune_heads` (it prunes a live `nn.Module` and rebuilds the config/head count, giving no control over the fused interleaved row order and no guarantee of bit-exactness); `mergekit` (layer/model-level merging, no intra-tensor axis slicing); `peft`, `torch-state-bridge` (key rewriting, not value slicing) — none address slicing inside a tensor axis.
- **Approximate time spent:** ~5 minutes.

## Checks enforced by the run
Before writing, `solution.py` asserts the three required layer-0 shapes and the 244-tensor count, plus the same three shapes on all 16 layers and per-key dtype preservation; any failure raises and no output is written. After writing, an independent re-read confirmed the key set, dtypes, the exact kept slices for the head-bearing tensors, and bit-exact equality for all other tensors.
