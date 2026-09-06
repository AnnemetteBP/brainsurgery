## Participant self-report

- Final artifact path: `out/T3/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The bfloat16 pattern must match only the four projection weights per layer
    (`attn.c_attn.weight`, `attn.c_proj.weight`, `mlp.c_fc.weight`,
    `mlp.c_proj.weight`) and not the biases or `attn.bias` buffer, so I anchored
    the regex to `\.weight$` on those exact submodule names rather than a
    loose `.*weight` pattern.
  - `h.<i>.attn.bias` is a non-parameter causal-mask buffer, not a weight; it
    had to be dropped entirely rather than cast or kept.
  - Sharding needs greedy bin-packing against a 64 MiB tensor-data budget per
    shard, with the oversized `wte.weight` (154 MB) forced into its own shard
    since it alone exceeds the limit.
- Anything in the task text or documentation that was unclear: none; the spec
  gave exact tensor names, shapes, and thresholds.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes to write and run the
  script once.
