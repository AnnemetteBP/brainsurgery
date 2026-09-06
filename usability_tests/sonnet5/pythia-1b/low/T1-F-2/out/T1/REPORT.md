## Participant self-report

- Final artifact path: `out/T1/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none; the only care needed
  was building an explicit old->new index remap up front so renumbering
  can't collide (survivors enumerated in original order, dropped blocks
  excluded, gaps closed).
- Anything in the task text or documentation that was unclear: no
- Tools used (condition F): `safetensors` 0.5.3 (`load_file`/`save_file`) in
  a plain Python script; no merge tool needed since this is a single-checkpoint
  key rewrite + drop, not a multi-model merge (mergekit) or adapter merge
  (peft), and torch-state-bridge's regex-capture rules add no safety over an
  explicit dict-based remap for a fixed, small 16->12 mapping.
- Approximate time spent, if you can tell: a few minutes
