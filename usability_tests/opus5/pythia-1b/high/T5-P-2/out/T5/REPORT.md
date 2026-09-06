# T5 — Participant self-report (condition P)

- **Final artifact path:** `out/T5/solution.py` (output: `out/T5/model-0000{1..4}-of-00004.safetensors`
  + `out/T5/model.safetensors.index.json`, 244 tensors, total_size 2,090,673,184 bytes)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the first execution passed
  all checks (16 pairs merged, 228 tensors bit-identical to the base, 4 shards each within
  the 512 MiB budget).
- **Pitfalls or surprises you hit (one line each):**
  - PEFT name mapping: adapter keys carry the `base_model.model.` prefix and end in
    `.lora_A.weight` / `.lora_B.weight`, so the base name is the stem plus `.weight`.
  - `fan_in_fan_out` is false here, so `B @ A` is `[out, in]` and is added untransposed; I
    still implemented the transpose branch off the config rather than hardcoding the layout.
  - The base contains non-float buffers that must survive untouched: 16 `attention.bias`
    tensors are `uint8 [1,1,2048,2048]` (4 MiB each) and `attention.masked_bias` is a 0-dim
    fp16 scalar. Sharding by byte size has to account for the uint8 masks, and the 0-dim
    scalar has to round-trip through `save_file`.
  - Sharding is by *tensor payload*, not file size: my shards are up to 529,611,484 bytes on
    disk (headers included) while the tensor data stays under 536,870,912.
  - `save_file` rejects tensors that share storage, so I materialised each shard with
    `.contiguous()` copies; loading the base with `load_file` already gives distinct storages.
- **Anything in the task text or documentation that was unclear:**
  - The sharding note says "A single tensor larger than that (here `gpt_neox.embed_in.weight`
    and `embed_out.weight`, 206 MB each) is stored alone in its own shard", but 206 MB is
    *below* the stated 536,870,912-byte limit, so no tensor in this checkpoint triggers the
    oversized rule. I implemented the numeric rule as written (oversized tensor ⇒ own shard,
    which never fires here) and greedy-filled otherwise; the two embedding tensors therefore
    share shard 1 (412 MB total). If the hidden reference instead isolates them, that is the
    one place my layout could differ.
  - The task fixes the shard *rules* but not a shard *order*, so a `weight_map` compared
    exactly against a reference would depend on iteration order. I used sorted key order for
    determinism; the shard count (4) and per-shard budgets satisfy the stated rules either way.
  - "No intermediate tensor appears in the output" is unambiguous for tensors; I read it as
    saying nothing about extra non-tensor files, and wrote only the shards and the index
    (no `config.json` / tokenizer copies).
- **Tools used (condition F):** n/a — condition P, only `torch` 2.14.0 and `safetensors` 0.5.3.
- **Approximate time spent, if you can tell:** ~5 minutes total; the script itself runs in
  ~10 s wall clock on CPU.
