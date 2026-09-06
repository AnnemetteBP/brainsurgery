# Participant self-report

- Final artifact path: `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to enumerate the 4 projection suffixes per layer explicitly rather
    than regex on `.*weight`, since that would also catch `wte.weight`,
    `wpe.weight`, and the layer-norm weights.
  - `h.<i>.attn.bias` (the causal-mask buffer) shares the `.bias` suffix
    naming pattern with real parameter biases like `attn.c_attn.bias`, so it
    had to be excluded by exact key match, not by suffix.
  - Sharding required special-casing `wte.weight` (154 MB) as its own shard
    since it alone exceeds the 64 MiB budget; a naive greedy packer without
    this case would either split a tensor (unsupported) or blow the budget.
- Anything in the task text or documentation that was unclear: none; the
  per-layer tensor names, shapes, and shard-size budget were fully specified.
- Tools used (condition F): `torch` 2.14.0 (`.to(torch.bfloat16)` cast) and
  `safetensors` 0.5.3 (`safe_open`/`save_file`) directly, no higher-level
  toolkit. Chosen over `transformers` dtype-export or `mergekit` because
  this task needs exact, name-level control over which 48 tensors are cast,
  which 12 buffers are dropped, and a specific greedy shard-packing rule
  (oversized tensor gets its own shard) — a plain script against the two
  lowest-level, already-available libraries was the most direct and
  auditable route, and let the required checks run as plain asserts before
  any file was written.
- Approximate time spent, if you can tell: a few minutes (single script,
  single successful execution).
