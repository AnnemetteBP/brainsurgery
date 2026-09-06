# Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The intended cast pattern (`h.<i>.(attn.c_attn|attn.c_proj|mlp.c_fc|mlp.c_proj).weight`)
    could easily over-match a broad `.*weight` regex onto embeddings/norms, so I anchored it to
    the four named projections per layer instead of a generic suffix match.
  - `attn.c_attn.bias` and `attn.c_proj.bias` are real parameter biases that must stay float32
    and must not be deleted; only the exact `h.<i>.attn.bias` causal-mask buffer is dropped, so
    the delete pattern is anchored with `attn\.bias` (no `c_attn`/`c_proj` prefix) to avoid
    catching those biases.
  - `cast_` (in place) was the right choice over `cast` (creates a new tensor) since the task
    requires tensor names to stay unchanged.
  - To pin the bfloat16 count at exactly 48 (not just "at least 48"), I added a negative-lookahead
    dtype assertion that everything *outside* the projection pattern is float32, in addition to
    checking the projection pattern itself is all bfloat16 with count 48.
  - Shard budget accounting: this tool's shard-size units are binary (`64MB` = 64 * 1024 * 1024
    bytes), matching the task's 64 MiB requirement directly, so no unit conversion was needed.
    `wte.weight` (154 MB) correctly landed alone in its own oversized shard per the documented
    "single tensor larger than budget" rule.
- Anything in the task text or documentation that was unclear: none; the README's sharding
  section and the `assert`/`cast_`/`delete` help text covered everything needed.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes of plan authoring, one execution.
