# T5 run record — condition P (Python / PyTorch)

## Participant self-report

- **Final artifact path:** `out/T5/solution.py` (output checkpoint in `out/T5/`:
  10 shard files + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single
  execution passed all pre-write and post-write checks.
- **Pitfalls or surprises you hit (one line each):**
  - The task text says `model.embed_tokens.weight` and `lm_head.weight` (412 MB)
    are "a single tensor larger than" the 512 MiB budget, but 412 MB is *below*
    536,870,912 bytes, so a plain greedy packer would not put them alone; I
    implemented an explicit rule — a tensor above half the budget gets its own
    shard — to produce the layout the task describes.
  - The base key order is alphabetical (that is what safetensors writes), and one
    OLMo layer is exactly 268,435,456 bytes, so exactly two layers fill one 512 MiB
    shard; the budget is clearly designed for that, which confirmed the shard plan
    (2 solo shards + 8 shards of 2 layers = 10).
  - PEFT name prefix is doubled: `base_model.model.` + `model.layers.<i>...`, so the
    base name is recovered by stripping only the first `base_model.model.` and the
    `.lora_{A,B}.weight` suffix; I matched it with a regex that also tolerates the
    `lora_A.<adapter_name>.weight` variant.
  - Base inputs are read-only and safetensors hands back mmap-backed tensors, so the
    merge has to be out-of-place (`W + delta`), not `W += delta`.
  - `fan_in_fan_out = false` and both the base and the factors use `[out, in]`, so
    `B @ A` is added untransposed; I still branched on the config flag rather than
    hard-coding it, and asserted the delta shape equals the base shape.
- **Anything in the task text or documentation that was unclear:**
  - The shard rule sentence quoted above is self-contradictory (412 MB is not larger
    than 512 MiB). I followed the stated *outcome* (both big tensors alone) rather
    than the literal threshold; a pure greedy packer would have put four layer-0
    attention tensors next to `model.embed_tokens.weight`.
  - The task lists only "shard files plus an index file" as the output, so I did not
    copy `config.json` / tokenizer files into `out/T5/`; a deployable checkpoint
    directory would normally need them.
  - `adapter_config.json` lists `target_modules` as `["q_proj", "v_proj"]` while
    TASK.md quotes `["self_attn.q_proj", "self_attn.v_proj"]`; harmless here since I
    derive targets from the adapter tensor names, not from `target_modules`.
- **Tools used (condition F):** n/a — condition P, only torch 2.14.0 and
  safetensors 0.5.3 (numpy 2.5.2 present but unused).
- **Approximate time spent, if you can tell:** ~10 minutes, of which the script run
  itself was 9.4 s wall clock (including reading the base twice for verification).

## Notes on the checks

Before writing: 64 adapter tensors resolved to exactly 32 complete A/B pairs, every
pair mapped to an existing base tensor, no `lora_` name in the planned output,
`model.layers.0.self_attn.q_proj.weight` present with shape `[2048, 2048]`, and
exactly 114 planned tensors matching the base key set.

After writing, the script reopens every shard and verifies: index/shard agreement in
both directions, no tensor in two shards, no stray `.safetensors` file, per-shard
tensor data ≤ 536,870,912 bytes, shapes and dtypes equal to the base, the 32 merged
tensors bit-equal to `base + 2 * B @ A` (relative Frobenius error 0), and the other
82 tensors bit-exact against the base.
