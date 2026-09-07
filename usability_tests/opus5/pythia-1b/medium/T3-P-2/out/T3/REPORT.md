# T3 participant self-report (condition P, pythia-1b)

- **Final artifact path:** `out/T3/solution.py` (output: `out/T3/model-0000N-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The obvious `.*weight` pattern would have caught `gpt_neox.embed_in.weight`, `embed_out.weight` and all layer-norm weights, so I anchored a regex on the four exact projection suffixes per layer instead.
  - `attention.bias` is a uint8 causal-mask buffer, not a parameter — deleting by name rather than by dtype was needed, and `masked_bias` is a 0-d scalar whose size is easy to mishandle.
  - Upcasting to float32 doubles the embedding tensors to 412 MB each, well over the 256 MiB shard budget, so the greedy packer needs an explicit "oversized tensor gets its own shard" branch rather than just closing the current shard.
  - Loaded lazily through `safe_open` and wrote shard by shard so the full float32 state dict (2.4 GB) is never resident at once.
  - `save_file` rejects non-contiguous tensors, so each cast result is `.contiguous()`.
- **Anything in the task text or documentation that was unclear:**
  - The shard *file naming* scheme is not specified; I used the HuggingFace convention `model-0000i-of-0000n.safetensors`.
  - The iteration order used for packing is not specified either, which determines the exact shard assignment; I used the input file's key order from `safe_open`. If the hidden reference packs in a different order the per-shard grouping will differ even though every stated rule holds.
  - Whether `metadata.total_size` is required in the index is not stated; I included it (float32/bfloat16 byte total, 2,436,513,792).
- **Tools used (condition F):** n/a — condition P (torch 2.14.0, safetensors 0.5.3 only).
- **Approximate time spent, if you can tell:** a few minutes; one pass to read the task and inputs, one to write and run the script.
