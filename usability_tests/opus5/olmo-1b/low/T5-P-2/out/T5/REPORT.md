# T5 participant self-report

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - PEFT name prefix: adapter keys carry `base_model.model.` on top of the base `model.layers.…` names, so the mapping is a prefix strip plus `.weight`, not a plain rename.
  - `fan_in_fan_out = false` with base weights in `[out, in]` layout means `B @ A` is added directly; a transpose here would have been silently shape-valid (2048x2048) and wrong, so I asserted on the config flag instead of assuming.
  - The two 412 MB tensors are actually under the 512 MiB budget, so greedy packing in sorted-name order already keeps every shard legal; the "alone in its own shard" clause only bites if a tensor really exceeds the cap, which I handled with an explicit flush.
  - `adapter_config.json` lists `target_modules` as `["q_proj", "v_proj"]`, not the dotted `self_attn.q_proj` form quoted in the task text; I derived base names from the adapter keys themselves rather than from that list.
- Anything in the task text or documentation that was unclear:
  - The shard file naming convention and shard ordering are not specified; I used the HF convention `model-000NN-of-000MM.safetensors` with tensors in sorted name order.
  - "412 MB each" vs a 512 MiB cap mixes decimal and binary units and reads as if those tensors exceed the budget, which they do not.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: ~5 minutes
