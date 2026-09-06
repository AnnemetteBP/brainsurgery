# T5 self-report (condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`,
  4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution
  succeeded and all four required checks passed.
- Pitfalls or surprises you hit (one line each):
  - PEFT name prefix: adapter keys carry `base_model.model.` and the
    `.lora_A/.lora_B` suffix, so mapping to base names is a strip on both ends.
  - `adapter_config.json` lists `target_modules: ["query_key_value"]` while
    TASK.md says `attention.query_key_value`; I derived base names from the
    adapter tensor names instead of from `target_modules`, which avoids the
    discrepancy.
  - The task text calls the two embedding tensors "larger than 512 MiB", but
    each is 206 MB, so ordinary greedy packing applies and no tensor needed a
    dedicated shard; I kept a guard that only trips for a genuinely oversized
    single tensor.
  - safetensors rejects non-contiguous tensors, so shards are saved with
    `.contiguous()`.
- Anything in the task text or documentation that was unclear:
  - The shard file naming and whether `metadata.total_size` is required in the
    index are not specified; I used the HuggingFace convention
    (`model-0000i-of-0000n.safetensors`, plus `total_size`).
  - The "single tensor larger than 512 MiB" sentence conflicts with the stated
    206 MB embedding size (see above).
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: ~5 minutes.
