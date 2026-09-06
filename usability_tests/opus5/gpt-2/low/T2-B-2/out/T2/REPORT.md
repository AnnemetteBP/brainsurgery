# T2 self-report (condition B)

- Final artifact path: `out/T2/model.safetensors` (plan: `out/T2/plan.yaml`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1 — `crash`: YAML ScannerError, the regex targets with `\.` were emitted inside double quotes, so PyYAML rejected `\.` as an unknown escape; fixed by switching those scalars to single quotes.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` ref to resolve to exactly one tensor, so no regex fan-out — the plan is explicit per layer (12 layers x 3 tensors = 36 concats, 96 sliced copies).
  - Destinations must not already exist, so each original tensor has to be `delete`d before the concatenated replacement can be written back under the same name.
  - Escaping: `from`/`to` in `copy`/`concat` are used as literal-ish names here, but `delete`/`assert` targets are regexes, so dots need escaping there (unescaped dots would also match e.g. nothing extra here, but `tmp\..*` needed it).
  - Conv1D `[in, out]` layout: head 5 is a column block in `c_attn.weight` (three times, once per q/k/v segment) but a row block in `c_proj.weight`; getting the axis right per tensor was the actual work.
- Anything in the task text or documentation that was unclear: the README documents slicing syntax but not whether sliced sources can be combined with regex fan-out in `copy`; I avoided the question by emitting explicit refs.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes.
