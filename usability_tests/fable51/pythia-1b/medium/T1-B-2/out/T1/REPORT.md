# T1 self-report (condition B, Pythia-1B)

- Final artifact path: `out/T1/plan.yaml` (output `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1 (the plan). One additional assert-only verification plan written to `/tmp` was run afterwards to compare the output against the input block by block; it wrote nothing.
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - `move` refuses an existing destination, so renumbering in ascending original order (3->2, 4->3, ..., 15->11) is collision-free and any wrong order would abort rather than overwrite.
  - Structured patterns (`[gpt_neox, layers, "3", "*rest"]`) move all 15 tensors of a block, including the `attention.bias` mask buffer and `rotary_emb.inv_freq`, in one transform; the regex form would have needed care with the dotted names.
  - Regex dots must be escaped and indices grouped (`(2|6|10|14)`) so `layers.1` does not also match `layers.10`.
- Anything in the task text or documentation that was unclear:
  - The doc pack does not state whether `move` with a structured pattern is atomic across all matches or per tensor; since it refuses existing destinations, ordering makes this moot here.
  - The task's first required check says "blocks 12, 13, 14, 15" while the removed blocks are 2, 6, 10, 14; I read it as "no index >= 12 remains after renumbering" and asserted that.
- Tools used (condition F): n/a.
- Approximate time spent, if you can tell: about 2 minutes.
