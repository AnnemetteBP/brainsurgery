# T5 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T5/plan.yaml` (output checkpoint in `out/T5/`,
  5 shards + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):**
  - None; the single execution succeeded and all asserts passed.

- **Pitfalls or surprises you hit (one line each):**
  - Output alias inference: with two inputs the plan must write to exactly one
    alias, so every intermediate had to be created on `base::` (not on `lora::`)
    to avoid `cannot infer output model uniquely`.
  - `fan_in_fan_out = true` plus the Conv1D `[in, out]` base layout means the
    product `B @ A` is `[2304, 768]` and must be transposed; I used
    `permute: { order: [1, 0] }` and asserted the `[768, 2304]` shape after it,
    so a wrong orientation would have failed loudly rather than silently.
  - Intermediates had to be named so they cannot be caught by the cleanup or
    the final checks by accident: regexes are full-match, so
    `h\.(\d+)\.attn\.c_attn\.mrgdelta` deletes only `mrgdelta` and leaves
    `mrgdeltat` / `mrgdeltas` alone, and I deliberately avoided `lora` in the
    intermediate names.
  - Sharding needed no manual work: `shard: 100MB` is binary (104,857,600
    bytes of tensor data) and `wte.weight` (154 MB) is automatically written
    alone in its own shard.
  - For the "no `lora_` in the output" check I used
    `assert: { not: { exists: ... } }` rather than `count: { is: 0 }`, since I
    was not sure a zero-match reference is accepted by `count`.

- **Anything in the task text or documentation that was unclear:**
  - The task lists `target_modules = ["attn.c_attn"]` while
    `inputs/lora/adapter_config.json` says `["c_attn"]`; the adapter tensor
    names resolve the ambiguity, so it did not matter.
  - The docs do not spell out how `matmul` pairs `from_a` with `from_b` when
    both are patterns; the interfaces reference's note that ternary transforms
    use "the same capture-based rewrite model across `from_a`, `from_b` and
    `to`" turned out to be the answer (`from_a` drives the matches, `from_b`
    and `to` are rewrites of each match) but an explicit `matmul` example with
    captures would have made that faster to confirm.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `docpack/help.txt` and the example plan; the plan run itself took ~11 s.

## What the plan does

```
W_i += (lora_alpha / r) * (B_i @ A_i).T ,  alpha/r = 32/16 = 2,  i = 0..11
```

as `matmul` -> `permute [1,0]` -> `scale 2.0` -> `add_`, then `delete` of the
three intermediate families, all on the `base` alias, written to `out/T5/`
with `shard: 100MB`.

Required checks, all present as `assert` transforms in the plan:

| Check | Implementation |
|---|---|
| exactly 12 adapter pairs found and merged | `count` on `lora_A` (12), `lora_B` (12), and on the 12 scaled deltas actually added |
| no `lora_` name in the output | `not: { exists: 'base::.*lora_.*' }` (plus `not: exists` for the intermediates) |
| `h.0.attn.c_attn.weight` still `[768, 2304]` | `shape` assert, before and after the merge |
| output has exactly 160 tensors | `count: { of: 'base::.*', is: 160 }` |

## Result of the run

160 tensors in `model.safetensors.index.json`, 5 shards, `wte.weight` alone in
its own shard, no `lora_` or intermediate names; 148 tensors byte-identical to
the base and exactly the 12 `h.<i>.attn.c_attn.weight` changed.
