# T5 participant self-report

- **Final artifact path:** `out/T5/solution.py` (output checkpoint in `out/T5/`,
  10 shards plus `model.safetensors.index.json`).
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - PEFT name prefix: adapter keys carry `base_model.model.` in front of the base
    name (`base_model.model.model.layers.<i>....`), so the mapping needs the double
    `model.model` stripped to exactly one `model.` — easy to strip one too many.
  - Scaling is `lora_alpha / r = 2`, not 1 and not `alpha` alone.
  - `fan_in_fan_out = false` means base weights are already `[out, in]`, matching
    `B @ A`, so no transpose; I still read the flag and transpose conditionally
    rather than hardcoding.
  - Shard budget: the task calls `model.embed_tokens.weight` / `lm_head.weight`
    "larger than" 512 MiB, but at 50304x2048 float32 they are 412,090,368 bytes,
    i.e. *under* the 536,870,912-byte budget. Greedy packing therefore pairs
    `model.embed_tokens.weight` with one 67 MB tensor (479 MB total, within
    budget) and leaves `lm_head.weight` alone. I followed the stated hard rule
    (no shard over 512 MiB of tensor data; a single oversized tensor alone)
    rather than the parenthetical's assumption.
  - `save_file` needs contiguous tensors; the merged weights are made contiguous
    explicitly after the add.
- **Anything in the task text or documentation that was unclear:**
  - The 412 MB "larger than 512 MiB" claim above is self-contradictory, and it
    leaves open whether the reference forces those two tensors alone in a shard.
  - The task does not say whether non-tensor files (`config.json`, tokenizer)
    should be copied into `out/T5/`; I wrote only shards and the index, since the
    required result lists exactly those.
  - Shard file naming is not specified; I used the HF convention
    `model-000NN-of-000NN.safetensors`.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent:** ~10 minutes.
