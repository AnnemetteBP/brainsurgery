## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the run succeeded on the first execution (an earlier attempt to run without saving used the wrong `inputs:` mapping shape, but was caught by reading `docpack/README.md` before invoking `brainsurgery`, so it isn't counted as an execution).
- Pitfalls or surprises you hit (one line each):
  - `inputs:` must be a list of `alias::path` strings, not a mapping — confirmed against the README before running.
  - Used `cast_` (in-place) rather than `cast` since tensor names must not change and `cast` requires a new destination name.
  - A single regex `h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight` was needed to hit exactly the 48 projection weights without also matching biases, embeddings, or `attn.bias`.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition B, plan-only).
- Approximate time spent, if you can tell: a few minutes.
