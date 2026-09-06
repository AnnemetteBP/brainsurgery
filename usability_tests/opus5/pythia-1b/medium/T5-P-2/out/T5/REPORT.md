# T5 — Participant self-report (condition P)

- **Final artifact path:** `out/T5/solution.py` (output in `out/T5/`: 4 shards +
  `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - PEFT name prefix: adapter keys carry `base_model.model.` and `.lora_A/.lora_B.weight`, so the
    base name is `stem[len("base_model.model."):] + ".weight"`.
  - Scaling is `lora_alpha / r = 2`, not `1`, and the merge has to happen in float32 before the
    cast back to float16 or the fp16 accumulation loses precision.
  - `fan_in_fan_out = false` with an `nn.Linear` `[out, in]` base means `B @ A` is added as-is;
    I still branched on the config flag rather than hardcoding it.
  - The task text says `embed_in`/`embed_out` (206 MB each) are "larger than" the 512 MiB budget
    and get their own shard, but 206 MB is well under 536,870,912 bytes, so the plain greedy rule
    packs them together in shard 1. I implemented the general rule (oversized tensor alone,
    otherwise greedy fill) and every shard stays under the budget.
  - `save_file` rejects non-contiguous / shared storage, so tensors are written `.contiguous()`.
- **Anything in the task text or documentation that was unclear:**
  - The 206 MB "larger than 512 MiB" claim above is self-contradictory; it left the intended shard
    assignment for the two embedding tensors ambiguous (rule-conformant vs. one-per-shard).
  - The task text says the adapted module is `attention.query_key_value` while
    `adapter_config.json` lists `target_modules: ["query_key_value"]`; I derived targets from the
    adapter tensor names instead of from `target_modules`, so it did not matter.
- **Tools used (condition F):** n/a — condition P (torch 2.14.0, safetensors 0.5.3 only).
- **Approximate time spent, if you can tell:** ~5 minutes.
