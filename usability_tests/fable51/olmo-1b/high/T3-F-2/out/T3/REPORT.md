# T3 self-report (condition F, olmo-1b, repeat 2)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 10 shard files plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Avoided regex targeting entirely: the 112 bf16 names are an explicit allowlist built from `model.layers.{i}.{suffix}`, so `.*weight` overreach onto `embed_tokens` / `lm_head` cannot happen.
  - Each layer's 7 projections are exactly 128 MiB in bf16, so two layers fill a 256 MiB shard to the byte; the budget check uses `>` (not `>=`) so equal-to-budget shards are allowed as the task states ("at most").
  - `embed_tokens` and `lm_head` (412 MB each) exceed the budget and are each written alone in their own shard; the packer flushes the current shard before emitting them.
  - Loaded via `safetensors.safe_open`, so `lm_head` and `embed_tokens` are independent tensors and `save_file` does not reject shared memory.
- Anything in the task text or documentation that was unclear:
  - The grader mentions "sharding rules" but the exact packing order of the hidden reference is not specified; I used embed_tokens, then layers 0..15 in numeric order (q,k,v,o,gate,up,down), then lm_head, greedy fill. If the grader requires an identical shard assignment rather than rule compliance, that ordering choice is the only degree of freedom.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: `tensor.to(torch.bfloat16)` for RNE cast, `torch.equal` for value checks.
  - `safetensors` 0.5.3: `safe_open` to read the input shards, `save_file` to write output shards with `{"format": "pt"}` metadata.
  - `json` (stdlib): reading the input index and writing the output index.
  - Did not use `transformers` `save_pretrained`: it applies one dtype to the whole model and its shard size is a file-size heuristic, so mixed precision plus the exact 256 MiB data budget with oversized tensors alone is simpler and more auditable as a direct script.
- Approximate time spent, if you can tell: about 3 minutes (script run ~6 s).
