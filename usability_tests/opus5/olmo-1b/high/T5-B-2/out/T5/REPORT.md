# T5 self-report (condition B, BrainSurgery plan)

- **Final artifact path:** `out/T5/plan.yaml` (output checkpoint in `out/T5/`).
  `out/T5/check.yaml` is an extra, independent verification plan; it writes no
  checkpoint and is not part of the solution.

- **Number of times you executed the script or plan:** 1 execution of
  `out/T5/plan.yaml`. Two additional runs of separate read-only plans:
  `out/T5/check.yaml` (post-hoc verification) and a throwaway
  `/tmp/cnt.yaml` (confirming the negative-lookahead pattern in the
  verification plan matched 82 of the 114 tensors rather than none).

- **Which executions failed, and why (one line each):** none; the plan passed
  on the first execution.

- **Pitfalls or surprises you hit (one line each):**
  - Output alias inference: `matmul`/`scale_`/`delete` all count as writes, so
    the intermediate `B @ A` products had to be created in the `base` alias
    (not in the `lora` alias) or the run would fail with
    "cannot infer output model uniquely".
  - The intermediate name had to avoid the substring `lora_`, since one of the
    required checks asserts no output tensor name contains it; I used the
    suffix `.merged_delta` and deleted those 32 tensors before the write.
  - Source references are full-match regexes but destination references are
    *rewrites* (dots stay unescaped there, backrefs `\1`/`\2` are substituted),
    so `from_a`/`to` in the same transform are written differently; this is what
    maps `base_model.model.model.layers.<i>.<m>.lora_B.weight` onto
    `model.layers.<i>.<m>.weight`.
  - `matmul` requires the destination to be absent while `add_` requires it to
    be present, so the order matmul -> scale_ -> add_ -> delete is forced.
  - Shard budget units are binary: `shard: 512MB` is 512 x 1024 x 1024 =
    536,870,912 bytes of tensor data, which is exactly the task's limit; the
    resulting 10 shards each hold at most that many bytes (file sizes exceed it
    only by the safetensors header).

- **Anything in the task text or documentation that was unclear:**
  - The task says `model.embed_tokens.weight` and `lm_head.weight` (412 MB
    each) are "larger than" the 512 MiB budget and therefore stored alone. They
    are not: 412 MB is ~393 MiB, below the budget. With the documented greedy
    packing in state-dict order, `lm_head.weight` does end up alone in shard 1,
    but `model.embed_tokens.weight` shares shard 2 with one 64 MiB tensor. I
    kept the tool's documented packing rather than forcing one tensor per shard.
  - The doc pack README's documentation links point at an absolute path from
    the maintainer's machine (`/Users/petersk/...`), so the referenced specs are
    not reachable from the sandbox; `help.txt` and
    `interfaces-reference.md` covered everything needed.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** roughly 10 minutes, of which the
  plan execution was ~14 s and the verification plan ~1 min.

## What the plan does

1. Preflight asserts: 114 base tensors, 32 `lora_A` and 32 `lora_B` tensors,
   shapes `[16, 2048]` / `[2048, 16]` / `[2048, 2048]`, all float32.
2. `matmul` pairs each `lora_B` with the `lora_A` of the same layer and module
   and writes `B @ A` (`[2048, 2048]`, float32) to
   `base::model.layers.<i>.<module>.merged_delta`. `fan_in_fan_out = false`
   means the factors already use the `[out, in]` layout, so no transpose.
3. Assert exactly 32 such products exist (the "32 pairs found and merged" check),
   with the right shape and dtype.
4. `scale_` them by `lora_alpha / r = 32 / 16 = 2`.
5. `add_` each product into the matching `model.layers.<i>.<module>.weight`.
6. `delete` all 32 intermediates.
7. Final asserts before writing: no name matching `.*lora_.*`, no leftover
   `merged_delta`, `model.layers.0.self_attn.q_proj.weight` still `[2048, 2048]`
   and float32, and exactly 114 tensors.
8. `output: { path: out/T5, format: safetensors, shard: 512MB }`.

## Independent verification (`check.yaml`, run after the fact)

Loading `out/T5`, `inputs/base` and the adapter side by side confirmed:
114 tensors and no `lora_` names; the 82 non-adapted tensors bit-identical to
the base; and for all 32 adapted weights, `out - base` equals `2 * (B @ A)`
within 1e-6 absolute (and is non-zero). The index file's `weight_map` covers
all 114 names over 10 shards, and `total_size` (5,119,148,032) matches the base.
