# T5 self-report (condition F, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T5/solution.py` (run as `.venv/bin/python out/T5/solution.py`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the first execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The task text says `model.embed_tokens.weight` / `lm_head.weight` are "larger than" the 512 MiB budget and therefore alone in a shard, but each is 412,090,368 bytes, i.e. *below* the budget; plain greedy packing puts `lm_head` alone anyway (the next tensor would overflow) while `embed_tokens` shares a shard with one 64 MiB tensor and still stays under the limit.
  - PEFT key prefix is triple: `base_model.model.` + `model.layers.<i>...`, so a naive `removeprefix("base_model.model.")` is exactly right but easy to over- or under-strip; I matched with an anchored regex instead.
  - `fan_in_fan_out=false` plus both base and adapter in `[out, in]` means `B @ A` is added untransposed; the config is read and the script refuses to run if it is ever `true`.
  - `adapter_config.json` lists `target_modules` as `["q_proj", "v_proj"]`, not the `self_attn.`-qualified names quoted in TASK.md; irrelevant here because the merge is driven by the adapter tensor names, not by that list.
- **Anything in the task text or documentation that was unclear:** the "stored alone in its own shard" sentence (see above) — it describes a rule for tensors exceeding the budget, but neither of the named tensors does. I implemented the stated numeric rule (no shard's tensor data exceeds 536,870,912 bytes, a single oversized tensor would get its own shard) via greedy packing in the base index order.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — lazy per-tensor read of the base shards and the adapter, and `save_file` for the output shards.
  - `torch` 2.14.0 — the float32 `B @ A` matmul and the add.
  - No `peft` / `transformers` / `mergekit`. `peft.merge_and_unload` would require materialising the model through `transformers`, and `save_pretrained` gives no control over the exact shard-packing rule and index this task grades (its `max_shard_size` uses a different accounting and it may tie `lm_head` to `embed_tokens`). A ~120-line script over safetensors does the closed-form update directly on the files, which is what the task asks for, and lets the four required checks run before a single byte is written.
- **Approximate time spent, if you can tell:** ~5 minutes.

## Required checks (all enforced in `solution.py:main`, before any write)

| Check | Where | Result |
|---|---|---|
| exactly 32 adapter pairs found and merged | `merged != EXPECTED_PAIRS` | 32 |
| no output tensor name contains `lora_` | `offenders` list comprehension | none |
| `model.layers.0.self_attn.q_proj.weight` is `[2048, 2048]` | `probe` shape check | ok |
| output has exactly 114 tensors | `len(base) != EXPECTED_TENSORS` | 114 |

Additionally `merge()` fails on an incomplete A/B pair, on an adapter key that
does not target an existing base tensor, on a delta whose shape does not match
the base weight, and on `fan_in_fan_out=true`; the write loop fails if a
multi-tensor shard exceeds the budget.

## Output

10 shards, 5,119,148,032 bytes of tensor data, `model.safetensors.index.json`
mapping all 114 names. Verified independently after the run: every merged
weight matches `base + 2.0 * B @ A` to a relative Frobenius error < 1e-6, the
82 other tensors are bit-identical to the base, all dtypes are float32, and no
shard exceeds 512 MiB.
