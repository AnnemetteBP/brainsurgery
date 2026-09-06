# T3 participant self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T3/solution.py` (run with `python out/T3/solution.py`; writes shards and `model.safetensors.index.json` into `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The causal-mask buffer `h.<i>.attn.bias` shares the `.bias` suffix with real projection biases, so it was matched by an anchored full-name regex rather than a suffix.
  - Projection weights were matched with an anchored regex naming the four Conv1D modules explicitly, to avoid `.*weight` catching `wte`, `wpe` and layer norms.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget, so the sharder places oversized tensors alone in their own shard instead of failing.
- Anything in the task text or documentation that was unclear: shard file naming and the index `metadata` block are not specified; I used the HuggingFace convention `model-0000i-of-0000n.safetensors` and `metadata.total_size`. Shard assignment order (input key order, greedy) is also unspecified.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: dtype cast (`tensor.to(torch.bfloat16)`, round-to-nearest-even) and exact-equality checks.
  - `safetensors` 0.5.3: `load_file` / `save_file` for reading the input and writing each shard.
  - Plain Python `re` / `json` for name matching and the index file. I did not use `transformers.save_pretrained` because its `dtype` argument applies one dtype to the whole model, whereas this task needs a per-tensor mix; a direct script gives exact control over the key set and the shard budget.
- Approximate time spent, if you can tell: about 2 minutes.

Checks enforced in `solution.py` before any file is written: exactly 48 bfloat16 tensors, `h.0.attn.c_attn.weight` bfloat16, `wte.weight` float32, exactly 148 output tensors, exactly 12 buffers dropped, all float32 tensors bit-identical to the input. After writing, a separate verification pass reloaded the shards through the index and confirmed the key set, dtypes, per-shard byte budget, and bit-exact values against the input.
