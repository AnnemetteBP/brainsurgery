# T1 self-report (condition B, GPT-2 124M)

- Final artifact path: `out/T1/model.safetensors` (plan: `out/T1/plan.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collisions: moves are ordered ascending (3->2, 4->3, 6->4, ...) so each destination is already vacant; `move` refuses existing destinations, which turns any wrong ordering into a loud failure.
  - Regex references are full-match, so `h\.3\.(.*)` cannot bleed into `h.30.*` style names; dots were escaped in every pattern.
  - `count` was used for all cardinality checks (12 blocks / 160 tensors before, 9 blocks / 117 block tensors / 121 total after); `not: exists` proves blocks 9-11 and 2/5/8 are gone.
- Anything in the task text or documentation that was unclear:
  - The README's "Tensor references" section says regex is full-match but the `help` output does not; I relied on the README.
  - The CLI prints the executed-transform summary to stdout on success; there is no explicit "OK" line, so success is inferred from the absence of an error and the presence of the output file.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes.
