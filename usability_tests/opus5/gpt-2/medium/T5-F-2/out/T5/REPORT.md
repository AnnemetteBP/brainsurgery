# T5 self-report

- **Final artifact path:** `out/T5/solution.py`
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the first execution produced the output.
- **Pitfalls or surprises you hit:**
  - The base key set has no `transformer.` prefix and still contains the 12
    `h.<i>.attn.bias` causal-mask buffers, so a `transformers` +
    `peft.merge_and_unload` + `save_pretrained` round-trip would very likely
    not reproduce the required 160-name key set (mask buffers are no longer
    persistent in current transformers, and `lm_head`/`wte` tying would need
    care). I chose direct tensor surgery instead of the "plausible route".
  - Conv1D layout: the base weight is `[in, out] = [768, 2304]` while `B @ A`
    is `[out, in]`, so the delta must be transposed — driven off
    `fan_in_fan_out` read from `adapter_config.json` rather than hardcoded.
  - Adapter names carry a `base_model.model.` prefix that must be stripped to
    reach base names.
  - `wte.weight` is 154 MB, above the 100 MiB shard budget, so the packer needs
    an explicit "oversized tensor gets its own shard" branch rather than just
    greedy accumulation.
- **Anything in the task text or documentation that was unclear:**
  - The shard budget is specified but not shard *naming* or the packing order;
    I used the HF convention `model-{i:05d}-of-{n:05d}.safetensors` and greedy
    packing in the base file's key order.
  - `target_modules` in the real `adapter_config.json` is `["c_attn"]`, while
    TASK.md quotes `["attn.c_attn"]`. Irrelevant here because I derive targets
    from the adapter tensor names, not from `target_modules`.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — `safe_open` / `save_file` for zero-model-instantiation
    checkpoint read and sharded write; the only thing needed for exact control
    over the key set, dtypes and shard membership.
  - `torch` 2.14.0 — float32 matmul, transpose and add for the low-rank delta.
  - Considered and rejected: `peft` 0.20.0 `merge_and_unload` (needs a live
    GPT-2 module and a matching key namespace; export would perturb the key
    set) and `mergekit` 0.1.4 (task-arithmetic merging of full checkpoints, no
    low-rank factor + `fan_in_fan_out` handling).
- **Approximate time spent:** ~10 minutes.

## Checks enforced by the run

`solution.py` raises `SystemExit("CHECK FAILED: ...")` before writing if:
exactly 12 adapter pairs are not found (also: unmatched `lora_B`, adapter target
missing from the base, duplicate targets, leftover adapter tensors, rank
disagreement with `r`, delta/weight shape mismatch); any output name contains
`lora_`; `h.0.attn.c_attn.weight` is not `[768, 2304]` float32; the output does
not have exactly 160 tensors with the base key set. Per-shard it also fails if a
multi-tensor shard exceeds 104,857,600 bytes and if the `weight_map` does not
cover 160 tensors.

## Independent verification (separate from the solution script)

Reloading `out/T5` through the index: 160 tensors, no `lora_` names, key set
equal to the base, all shapes/dtypes equal, 148 tensors bit-identical to the
base, 12 merged weights with relative Frobenius error 0.0 against a
re-derived `W + 2 * (B @ A).T`, and every shard within budget (`wte.weight`
alone in shard 5).
