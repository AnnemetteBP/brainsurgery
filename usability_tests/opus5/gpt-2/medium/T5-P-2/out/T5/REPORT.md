# T5 (GPT-2 124M) — Condition P self-report

- **Final artifact path:** `out/T5/solution.py` (output checkpoint in `out/T5/`:
  5 shards `model-0000N-of-00005.safetensors` + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution passed all checks.
- **Pitfalls or surprises you hit (one line each):**
  - Conv1D layout: the base `h.<i>.attn.c_attn.weight` is `[in, out] = [768, 2304]` while `B @ A` is `[out, in]`, so `fan_in_fan_out = true` means the delta must be transposed before adding — I read the config rather than hard-coding the transpose.
  - PEFT name prefix `base_model.model.` has to be stripped to reach the base name, and `.lora_A/.lora_B` replaced by `.weight`.
  - The key order inside `model.safetensors` is not sorted and not the module order, so `wte.weight` (147 MiB) is not first; the greedy sharding loop needed an explicit "oversized tensor gets its own shard" branch instead of relying on it happening to start a fresh shard.
  - `(B @ A).T` is non-contiguous, so the merged result is `.contiguous()`-ed before saving to avoid a safetensors save error.
  - The 100 MiB budget counts tensor data only, not headers, so the budget check sums `numel * element_size` rather than file sizes.
- **Anything in the task text or documentation that was unclear:**
  - The task fixes the per-shard budget and the "oversized tensor alone" rule but not the shard *file naming* or the iteration order that determines which tensor lands in which shard; I used the HF convention (`model-{i:05d}-of-{n:05d}.safetensors`, greedy in checkpoint key order) and assumed grading checks the sharding rules rather than an exact shard assignment.
  - It was not stated whether `index.json` should carry `metadata.total_size`; I included it since HF readers expect it.
- **Tools used (condition F):** n/a — condition P (plain Python: `torch` 2.14.0, `safetensors` 0.5.3).
- **Approximate time spent, if you can tell:** ~10 minutes, most of it inspecting the safetensors header to size the shards before writing the script.
