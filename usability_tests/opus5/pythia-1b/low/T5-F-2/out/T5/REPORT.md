# T5 self-report (condition F)

- **Final artifact path:** `out/T5/solution.py` (output checkpoint in `out/T5/`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution passed all checks.
- **Pitfalls or surprises you hit (one line each):**
  - PEFT name prefix: adapter keys carry `base_model.model.` and `.lora_A/.lora_B`, so the base name is `stem.removeprefix("base_model.model.") + ".weight"`.
  - `fan_in_fan_out = false` with the `[out, in]` Linear layout means `B @ A` is added directly, no transpose; I assert on the config rather than assuming.
  - The task says the 206 MB embedding tensors are "larger than" the 512 MiB budget and must be alone in a shard, but 206 MB is well under 512 MiB; I followed the explicit numeric rule and used the standard greedy in-order packing (`huggingface_hub.split_torch_state_dict_into_shards`), which puts `gpt_neox.embed_in.weight` and `embed_out.weight` together in shard 1 (530 MB file, 512 MiB of tensor data). If the hidden reference instead isolates each embedding, that is the one place my output could differ.
  - safetensors rejects non-contiguous/shared tensors, so shards are saved with `.contiguous()`.
- **Anything in the task text or documentation that was unclear:** the sharding parenthetical above (contradicts its own threshold); also whether the index `metadata.total_size` is graded (I emit the standard one).
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — direct load/save of the checkpoint files, which is the whole point of doing this without instantiating a model.
  - `torch` 2.14.0 — the float32 matmul and the float16 cast back.
  - `huggingface_hub` (pinned) `split_torch_state_dict_into_shards` — the canonical HF sharding/index logic, so shard naming and `weight_map` match what a `save_pretrained` reference would produce.
  - Considered and rejected: `peft.merge_and_unload`, which requires materialising the model through `transformers` and would re-emit a full model rather than edit the checkpoint; the task explicitly asks to avoid instantiating the model. `mergekit` has no LoRA-fold path that is simpler than nine lines of matmul.
- **Approximate time spent:** ~5 minutes.

## Checks enforced by the run
Before writing: exactly 16 adapter pairs merged, no `lora_` name in the output, `gpt_neox.layers.0.attention.query_key_value.weight` still `[6144, 2048]`, exactly 244 tensors. Plus per-pair rank/shape assertions. After writing: every shard re-read, per-shard tensor bytes <= 512 MiB (or a single oversized tensor), no duplicate names, values equal to the in-memory result, and `weight_map` consistent with the shards.
