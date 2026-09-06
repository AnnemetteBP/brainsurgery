# T5 self-report (condition F, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T5/solution.py` (output checkpoint in `out/T5/`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single
  execution succeeded and all four required checks passed.
- **Pitfalls or surprises you hit (one line each):**
  - PEFT name prefixes: adapter keys carry `base_model.model.` in front of the
    base name (`base_model.model.model.layers.<i>...`), so exactly one prefix
    has to be stripped, not two.
  - `adapter_config.json` lists `target_modules` as `["q_proj", "v_proj"]`
    while TASK.md says `["self_attn.q_proj", "self_attn.v_proj"]`; I ignored
    `target_modules` entirely and drove the merge from the adapter tensor
    names, which avoids the discrepancy.
  - Shard budget: the two 412 MB tensors are *below* the 512 MiB budget, so
    HF's greedy packer puts `lm_head.weight` alone but pairs
    `model.embed_tokens.weight` with a following tensor (479 MB total). That
    still satisfies "at most 512 MiB per shard"; the "alone in its own shard"
    sentence only binds for tensors larger than the budget.
  - The base's own index is 2 shards at the default 5 GB limit, so the input
    sharding gives no hint about the required output sharding.
- **Anything in the task text or documentation that was unclear:** whether the
  shard file naming must follow `model-000NN-of-000MM.safetensors` (I used the
  HF convention), and the `target_modules` mismatch noted above.
- **Tools used (condition F):**
  - `torch` 2.14.0 — float32 matmul `B @ A` and the scaled add.
  - `safetensors` 0.5.3 — `load_file` / `save_file` for direct checkpoint I/O.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards` for the
    512 MiB greedy sharding and the `weight_map`, so the output follows the
    standard HF sharding rules rather than a hand-rolled packer.
  - Deliberately **not** `peft.merge_and_unload`: that route instantiates the
    full model through `transformers`, costs several GB of RAM and a dtype
    round-trip, and the task explicitly wants the merge done on the checkpoint
    files. A ~60-line state-dict script is smaller, faster and easier to make
    fail loudly.
- **Approximate time spent, if you can tell:** ~5 minutes.

## Verification performed

- In-script (fails before writing): 32 pairs merged, no `lora_` key in the
  output, `model.layers.0.self_attn.q_proj.weight` is `[2048, 2048]`, exactly
  114 tensors, per-tensor delta/base shape match, base dtype is float32, every
  written shard within the 512 MiB budget, and a read-back of the written
  shards covering the full key set.
- Post-hoc: exactly 32 tensors differ from the base (all `q_proj`/`v_proj`),
  the other 82 are bit-identical, and a spot-checked layer equals
  `W + 2 * (B @ A)` bit-exactly in float32.
