# Participant self-report: T3 (GPT-2 124M), condition B

- Final artifact path: `out/T3/plan.yaml` (output checkpoint: `out/T3/model-0000{1..4}-of-00004.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `h.<i>.attn.bias` (causal mask) shares the `.bias` suffix with real parameters, so it must be targeted by its exact path rather than a broad `.*bias` pattern.
  - `.*weight` would hit `wte`, `wpe` and layer norms; I used an explicit regex over the four projection module names.
  - "exactly 48 tensors are bfloat16" has no direct operator, so I expressed it as: the projection regex matches 48, all of them are bfloat16, and the complement (negative lookahead) is all float32.
- Anything in the task text or documentation that was unclear:
  - The README does not say whether `assert: dtype` checks every match of a pattern or only one; the installed source shows it checks every match, which I relied on.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes.
