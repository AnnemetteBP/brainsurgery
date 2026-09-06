# T5 self-report

- Final artifact path: `out/T5/solution.py` (output: `out/T5/model-0000{1..5}-of-00005.safetensors` + `out/T5/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - Conv1D `[in, out]` vs Linear `[out, in]`: the delta `B @ A` had to be transposed, driven off `fan_in_fan_out` in the adapter config rather than hardcoded.
  - Adapter keys carry a `base_model.model.` prefix that must be stripped to reach the base names.
  - Sharding with a 100 MiB budget: `wte.weight` (154 MB) exceeds the budget on its own, so the greedy packer has to flush before it and emit it alone rather than refuse.
- Anything in the task text or documentation that was unclear:
  - The shard file naming scheme is not specified; I used the HuggingFace convention `model-{i:05d}-of-{n:05d}.safetensors`.
  - Whether the index must carry `metadata.total_size` is unstated; I included it, as HF does.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, plus stdlib `json`. I deliberately did **not** use `peft.merge_and_unload`: that route requires instantiating a GPT-2 model, which the task explicitly wants to avoid, and it would not give control over the 100 MiB shard budget or the required pre-write checks. `transformers.save_pretrained` was also skipped because its sharding is driven by `max_shard_size` string parsing and it re-ties/renames keys, which risks diverging from the exact 160-name key set. A ~90-line script over safetensors does the name mapping, scaling, transpose and packing directly and fails loudly on each required check.
- Approximate time spent, if you can tell: ~5 minutes.

## Checks enforced by the run

Before writing: exactly 12 adapter pairs merged; rank agrees with `r`; delta shape matches the base tensor; no `lora_` name in the state dict; `h.0.attn.c_attn.weight` is `[768, 2304]` float32; exactly 160 tensors; no multi-tensor shard over 100 MiB. After writing: shards are re-read and the key set is compared against the base file, count re-checked, `lora_` re-checked. Any failure raises `SystemExit("CHECK FAILED: ...")`.

## Verification performed

Re-loading the output: 160 tensors, 148 bit-identical to the base, exactly the 12 `h.<i>.attn.c_attn.weight` changed; spot-check of `h.3.attn.c_attn.weight` against `base + 2*(B@A).T` gives relative Frobenius error 0.0. `wte.weight` sits alone in shard 5.
