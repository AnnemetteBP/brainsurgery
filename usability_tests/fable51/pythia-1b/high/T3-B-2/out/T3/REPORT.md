# Participant self-report: T3 (Pythia-1B), condition B

- Final artifact path: `out/T3/plan.yaml` (output: `out/T3/model-0000{1..9}-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 succeeded.
- Pitfalls or surprises you hit (one line each):
  - There is no assert that counts tensors by dtype, so "exactly 64 bfloat16" is expressed as: the 64-projection regex matches exactly 64 tensors AND all of them are bfloat16 AND every tensor not matched by that regex (negative-lookahead regex) is float32.
  - Order of casts matters: upcast everything with `cast_: {target: '.*', to: float32}` first, then downcast the 64 projections to bfloat16; the reverse would upcast the projections back.
  - `attention.bias` (mask buffer) vs `attention.dense.bias` (parameter): full-match regex with escaped dots keeps them apart; a loose `.*bias` would delete parameters.
  - The float32 embeddings are 412 MB each, above the 256 MiB budget; the tool placed each alone in its own shard as the task expects.
- Anything in the task text or documentation that was unclear:
  - The README documents shard-size units (`256MB` = 256 x 1024 x 1024) and the oversized-tensor rule clearly; nothing blocking. The docs do not state explicitly whether `dtype:` in `assert` applies to every match of a pattern, but it does.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: roughly 5 minutes (reading docs, one plan run, verification of the index).
