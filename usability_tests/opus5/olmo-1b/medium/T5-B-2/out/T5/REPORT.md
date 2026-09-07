# T5 self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`,
  10 shards plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run
  passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - `output` writes exactly one alias and infers it from which alias the
    transforms write to, so the `B @ A` product had to be created *inside* the
    `base` alias (as scratch names `merge_delta.<i>.<module>`) rather than in
    the `lora` alias; touching the `lora` alias at all (even a `delete`) would
    have made the output alias ambiguous.
  - The scratch tensors therefore have to be `delete`d before writing, which
    the "no intermediate tensor in the output" requirement demands anyway.
  - PEFT name prefix: adapter keys carry `base_model.model.` in front of the
    base name, so the mapping is
    `base_model.model.model.layers.<i>.<m>.lora_{A,B}.weight` ->
    `model.layers.<i>.<m>.weight`; regex dots must be escaped on the match side.
  - `matmul` argument order matters: `from_a` = `lora_B` `[2048,16]`,
    `from_b` = `lora_A` `[16,2048]`, giving `[out, in]`, which matches the base
    `nn.Linear` layout, so `fan_in_fan_out = false` means no transpose.
  - Scaling is `lora_alpha / r = 32 / 16 = 2`, applied to the product with
    `scale_` before `add_`.
  - `shard: 512MB` is binary (536,870,912 bytes of tensor data), exactly the
    budget in the task; the writer packs greedily and puts
    `model.embed_tokens.weight` alone in shard 1.
- Anything in the task text or documentation that was unclear:
  - The task calls `model.embed_tokens.weight` / `lm_head.weight` "larger than"
    512 MiB, but at 412 MB (393 MiB) they are actually under the budget, so the
    "stored alone" rule is not what isolates them; greedy packing does.
  - The docs do not spell out that `assert`/`dump`/`diff` are the only
    transforms that do not count toward output-alias inference until the note
    in `interfaces-reference.md` section 3; that note was the key to the design.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes, most of it reading
  `docpack/help.txt` and the example plan; the run itself took 12 s.
