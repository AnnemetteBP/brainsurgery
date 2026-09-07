## Participant self-report

- Final artifact path: `out/T3/` (10 shards `model-0000X-of-00010.safetensors` + `model.safetensors.index.json`, plus `solution.py` and `run.sh` used to produce them)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the one run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The two embedding-family tensors (`model.embed_tokens.weight`, `lm_head.weight`) are each 412 MB, over the 256 MiB shard cap on their own, so the bin-packer needs an explicit "oversized tensor gets its own shard" branch rather than a uniform greedy pack.
  - The 256 MiB budget is tensor-data-only; safetensors file sizes on disk are a few KB larger per shard from the header, which is expected and not a bug.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — direct shard read/write with per-tensor dtype control; needed because the cast is name-pattern-scoped (112 of 114 tensors), which `transformers`' `save_pretrained(torch_dtype=...)` cannot express (it applies one dtype to the whole model, with no per-tensor-name override).
  - `torch` 2.14.0 — `Tensor.to(torch.bfloat16)` for the round-to-nearest-even cast; and for shape/dtype bookkeeping.
  - Plain Python (`re`, `json`) for the regex targeting of the 112 projection matrices and for writing the index file — no merge/adapter tool (`mergekit`, `peft`) was applicable since there is nothing to merge or adapt, and `torch-state-bridge`'s value is key-renaming, which this task doesn't need (names are unchanged).
- Approximate time spent, if you can tell: ~10 minutes.
