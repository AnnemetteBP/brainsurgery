# T3 self-report (condition B)

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - The obvious `.*weight` pattern would have caught `wte.weight`, `wpe.weight` and the layer norms, so I enumerated the four projection names explicitly in one regex with escaped dots.
  - The causal-mask buffers are named `h.<i>.attn.bias`, which looks like a parameter bias; the delete pattern had to be anchored to `attn.bias` exactly and not the `c_attn`/`c_proj` biases.
  - Shard budget units are binary in this tool, so `shard: 64MB` is exactly the 67,108,864-byte budget required; `wte.weight` (154 MB) landed alone in its own shard automatically.
- Anything in the task text or documentation that was unclear:
  - Whether `assert: dtype` over a multi-match reference checks every match or only the first; I hedged by also asserting the match `count: 48` and a single-tensor dtype check on `h.0.attn.c_attn.weight`.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes, mostly reading the README's output/sharding section.
