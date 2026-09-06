# T5 — LoRA adapter merge with sharded export (OLMo-1B-0724-hf), condition F

## Participant self-report

- **Final artifact path:** `out/T5/solution.py` (run as
  `.venv/bin/python out/T5/solution.py` from the sandbox root)
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  - Execution 1: produced a fully correct `out/T5/` (all checks passed, exit 0),
    but `write_sharded()` cleared the output directory with
    `shutil.rmtree(OUT_DIR)` and `OUT_DIR` is `out/T5/` — which is also where
    the rules say the artifact must live, so the script deleted its own source
    file as a side effect. Not a failed run by exit status; a self-destructive
    one.
  - Execution 2: same script with the cleanup narrowed to `*.safetensors` plus
    the index file. Passed, and the artifact survived.
- **Pitfalls or surprises you hit (one line each):**
  - PEFT name prefix: adapter keys carry `base_model.model.` on top of the base
    name (`base_model.model.model.layers.<i>...`), so the mapping is a prefix
    strip plus `.lora_{A,B}.weight` -> `.weight`, not a plain suffix rename.
  - `fan_in_fan_out = false` means base and factors share the `[out, in]`
    layout, so `B @ A` is added untransposed; I made the script assert the flag
    is false rather than silently handle both layouts.
  - TASK.md says a tensor larger than 512 MiB is stored alone, but the two
    named tensors are ~393 MiB each, i.e. under the budget; I implemented the
    stated budget rule (<= 536,870,912 bytes of tensor data per shard) and the
    greedy packer puts `lm_head.weight` alone in shard 1 anyway.
  - Self-inflicted: the artifact and the output checkpoint share a directory
    (`out/T5/`), so a "clean the output dir first" `rmtree` deletes the script
    that is running. Scoped the cleanup to the files the script itself writes.
  - `tie_word_embeddings` is false here, so `lm_head.weight` is a genuine 114th
    tensor — a `save_pretrained` route would have been at risk of dropping it
    on a tied model; the file-level route sidesteps that.
- **Anything in the task text or documentation that was unclear:**
  - The shard-size rule is stated as a bound on tensor payload, but the packing
    order (and hence the exact shard assignment) is not specified. I used
    sorted key order with the canonical HF packer, which satisfies the stated
    rules regardless.
  - Whether `config.json` / tokenizer files should be copied into `out/T5/`.
    The spec enumerates "shard files plus an index file", so I wrote only those.
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — read the base shards and the adapter, write the
    output shards. Direct file I/O, no model graph needed.
  - `torch` 2.14.0 — float32 `B @ A` matmul, scaling and addition, and the
    equality/norm checks.
  - `huggingface_hub` 1.16.1 (`split_torch_state_dict_into_shards`) — the
    canonical HF shard packer, driven with `max_shard_size = 512 MiB`, so the
    sharding and the `weight_map` follow HF conventions instead of a hand-rolled
    packer.
  - **Not used, deliberately:** `peft` 0.20.0 `merge_and_unload` and
    `transformers` `save_pretrained`. That route is the "obvious" one for this
    task, but it instantiates a 1B-parameter model to perform what is a
    32-tensor rank-16 update, and it routes the output through
    `save_pretrained`, whose key set depends on weight tying and whose dtype
    handling is another thing to pin down. The task explicitly wants the edit
    done on the checkpoint files; the file-level route is ~9 s end to end,
    keeps the 82 untouched tensors bit-identical by construction, and makes
    every required check a local assertion.
- **Approximate time spent, if you can tell:** ~15 minutes, of which each run
  is about 10 seconds.

## How the required checks are enforced

All four are `check()` calls that raise `CheckFailed` and exit non-zero. Each
runs twice: once on the in-memory state dict *before* anything is written, and
again in `verify_written()` on the state dict re-read from `out/T5/`.

| Required check | Where |
|---|---|
| exactly 32 adapter pairs found and merged | `load_adapter()` (pair count + 64 tensors), `merge()` return, `required_checks()` |
| no output tensor name contains `lora_` | `required_checks()` |
| `model.layers.0.self_attn.q_proj.weight` is `[2048, 2048]` | `required_checks()` |
| output has exactly 114 tensors | `required_checks()`, plus key-set equality with the base |

Additional guards: `fan_in_fan_out` must be false, `r > 0`, factor shapes must
agree with `r`, delta shape must equal base shape, all output tensors must be
float32, each merged tensor must actually differ from its pre-merge value, no
tensor may appear in two shards, each shard's tensor payload must be
<= 536,870,912 bytes, and the index `weight_map` must agree with the files on
disk.

## Result of the final execution

```
adapter: 32 pairs, scale = alpha/r = 2.0
verified 114 tensors across 10 shards
wrote out/T5: 114 tensors, 32 merged
```

10 shards, largest payload exactly 536,870,912 bytes; `lm_head.weight` alone in
shard 1. An independent re-check against `inputs/base` and the adapter (not
part of the script) confirmed: 114 tensors, key set identical to the base, no
`lora_` names, `model.layers.0.self_attn.q_proj.weight` at `[2048, 2048]`, the
82 unchanged tensors bit-exact, the 32 merged tensors at relative Frobenius
error 0.0 versus `base + 2 * B @ A`, all float32, every shard within the
512 MiB budget.
