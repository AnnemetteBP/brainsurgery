# T5 self-report (condition B)

- **Final artifact path:** `out/T5/plan.yaml` (output checkpoint in `out/T5/`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the first run succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - With two inputs there is no default alias, so every reference (including
    `assert` targets) needs an explicit `base::` / `lora::` prefix.
  - The output alias is inferred from what the transforms *write to*, so the
    `B @ A` products had to be created on the `base` alias, not on `lora`.
  - `matmul` creates new tensors, so the intermediate deltas live in the output
    alias and must be `delete`d before writing; I named them `bsdelta.<i>.<q|v>.weight`
    and asserted they are gone.
  - PEFT's `base_model.model.` prefix plus the extra `model.` means the adapter
    name is `base_model.model.model.layers.<i>...`; the rewrite side of a
    reference uses plain `\1`/`\2` backreferences with unescaped dots.
- **Anything in the task text or documentation that was unclear:**
  - The task says `model.embed_tokens.weight`/`lm_head.weight` (412 MB) are
    "larger than" the 512 MiB budget and get their own shard; they are actually
    smaller, so the tool's ordinary in-order packing applies. The written output
    has 10 shards, each at most 536,870,912 bytes of tensor data.
  - `shard: 512MB` is documented as binary (512 * 1024^2), which is what the task
    wants; worth stating explicitly since `MB` is decimal elsewhere.
- **Tools used (condition F):** n/a (condition B).
- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `docpack/README.md`, `help.txt` and the MoE example plan.

## Checks implemented in the plan

- `count` of `lora_A` and of `lora_B` matches = 32 each, and `count` of the 32
  computed deltas = 32 (i.e. exactly 32 pairs found and merged);
- `not: exists` for any name containing `lora_` (and for the intermediates);
- `shape` of `model.layers.0.self_attn.q_proj.weight` is `[2048, 2048]`;
- `count` of all output tensors = 114 (asserted before and after the merge).
