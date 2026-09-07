# T5 — Participant self-report

- **Final artifact path:** `out/T5/solution.py` (output checkpoint: `out/T5/model-0000{1..5}-of-00005.safetensors` + `out/T5/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution passed all checks.
- **Pitfalls or surprises you hit (one line each):**
  - Conv1D layout: base `h.<i>.attn.c_attn.weight` is `[in, out] = [768, 2304]` while `B @ A` is `[out, in]`, so the product must be transposed before adding; `fan_in_fan_out: true` in the adapter config is the signal for this.
  - PEFT name prefix `base_model.model.` has to be stripped, and `lora_A`/`lora_B` suffixes removed, to reach the base key.
  - Scale is `lora_alpha / r = 32 / 16 = 2`, not 1 and not `1/r`.
  - `wte.weight` is 154 MB, larger than the 100 MiB shard budget, so the packing loop needs an explicit "oversized tensor gets its own shard" branch rather than plain greedy accumulation.
  - Shard budget is tensor *data* only; the written files are a few KB larger because of the safetensors header, so any post-write size check has to sum tensor bytes, not `stat` the file.
  - `h.<i>.attn.bias` are 4 MB causal-mask buffers that also live in the checkpoint; they are ordinary tensors to copy, not weights to touch.
- **Anything in the task text or documentation that was unclear:**
  - The task fixes the shard budget and the oversized-tensor rule but not the shard ordering or file naming. I sorted tensor names alphabetically and used the HuggingFace `model-{i:05d}-of-{n:05d}.safetensors` convention; a different but equally valid order would produce a different partition into 5 shards.
  - `target_modules` is `["c_attn"]` in `adapter_config.json` while the task text says `attn.c_attn`; I keyed off the adapter tensor names instead of the config list, which sidesteps the discrepancy.
- **Tools used (condition F):** n/a — condition P (plain Python + torch 2.14.0 + safetensors 0.5.3).
- **Approximate time spent, if you can tell:** ~5 minutes: reading the safetensors headers directly out of the input files to confirm the exact key names and shapes, then one pass to write the script.

## What the script does

1. Reads `r`, `lora_alpha`, `fan_in_fan_out` from `adapter_config.json`; asserts `fan_in_fan_out` is true and computes `scale = alpha / r`.
2. Loads base (160 tensors) and adapter (24 tensors); pairs `lora_A`/`lora_B` per target after stripping `base_model.model.`, and fails if any pair is incomplete or the count is not 12.
3. For each target: `W += scale * (B @ A).T`, all in float32, with a shape check against the base tensor.
4. Required checks before writing: exactly 12 pairs merged, no `lora_` name in the output, `h.0.attn.c_attn.weight` is `[768, 2304]`, exactly 160 tensors and the same key set as the base.
5. Packs the 160 tensors greedily in sorted-name order into shards of at most 104,857,600 bytes of tensor data, giving any tensor above that limit a shard to itself (`wte.weight`), and writes the index with a complete `weight_map`.
6. Re-opens every written shard and verifies key uniqueness, shapes, dtypes, values against the merged state dict, the per-shard budget, and that `weight_map` matches what is actually on disk.

Result: 5 shards (41 / 41 / 45 / 32 / 1 tensors), 160 tensors, `total_size = 548,090,880` bytes.
