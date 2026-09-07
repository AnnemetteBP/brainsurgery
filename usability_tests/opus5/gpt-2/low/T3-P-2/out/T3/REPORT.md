# T3 self-report (condition P)

- Final artifact path: `out/T3/solution.py` (output in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks.
- Pitfalls or surprises you hit (one line each):
  - Over-broad targeting is the obvious trap: I anchored the cast regex to the four
    exact projection names per layer instead of anything like `.*weight`.
  - `h.<i>.attn.bias` is a causal-mask buffer, not a bias parameter, so name-based
    "keep all biases in fp32" logic would have wrongly kept it.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget, so the greedy packer must
    allow a single oversized tensor to occupy a shard alone.
- Anything in the task text or documentation that was unclear:
  - Shard file naming and tensor ordering are not specified; I used the HF convention
    `model-0000i-of-0000N.safetensors` and the input file's key order.
  - Whether the index needs `metadata.total_size` is unstated; I included it.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: a few minutes.
