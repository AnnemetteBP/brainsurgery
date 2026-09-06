# T5 self-report

- Final artifact path: `out/T5/solution.py` (output in `out/T5/`, 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Conv1D layout: base `h.<i>.attn.c_attn.weight` is `[in, out] = [768, 2304]`, so the `B @ A` product `[2304, 768]` has to be transposed (`fan_in_fan_out = true`) before adding.
  - PEFT name prefix: adapter keys carry `base_model.model.` in front of the base name and `.lora_A/.lora_B` behind it; both ends must be stripped/rebuilt.
  - Scaling is `lora_alpha / r = 2`, not `1`, and is applied to the product, not to a factor.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget on its own, so the budget check must exempt single-tensor shards; alphabetical key order conveniently puts it last, alone in shard 5.
  - Tensors were cloned + made contiguous before `save_file` to avoid safetensors rejecting shared/aliased storage.
- Anything in the task text or documentation that was unclear:
  - The shard file naming scheme is not specified; I used the HuggingFace convention `model-000NN-of-000NN.safetensors`.
  - "at most 100 MiB of tensor data" left the packing policy implicit; I used greedy sequential packing in key order, which is what `save_pretrained` does.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: a few minutes; one read of the inputs, one script, one run.
