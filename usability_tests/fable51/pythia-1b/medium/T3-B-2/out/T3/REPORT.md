# T3 (Pythia-1B), condition B: participant self-report

- Final artifact path: `out/T3/plan.yaml` (output checkpoint: `out/T3/model-0000{1..9}-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all assertions and wrote 196 tensors across 9 shards.
- Pitfalls or surprises you hit (one line each):
  - The two embedding matrices become 412 MB each once upcast to float32 (TASK.md quotes 206 MB, their float16 size); they still exceed the 256 MiB budget and land alone in shards 1 and 2 as required.
  - There is no assert operator that counts tensors by dtype, so "exactly 64 tensors are bfloat16" was expressed as: the 64-matrix pattern matches 64 tensors and they are all bfloat16, plus the complementary negative-lookahead pattern matches 132 tensors and they are all float32.
  - Ordering matters: cast everything to float32 with `.*` first, then narrow the 64 projection matrices to bfloat16; the buffers are deleted before the blanket cast so the uint8 masks are never touched.
- Anything in the task text or documentation that was unclear:
  - Whether `assert: dtype` with a multi-match `of` checks every match; the run confirmed it does (the combined `all` block passed with 64-match and 132-match patterns).
  - `assert.not` help says "Key metadata: unavailable", so its payload form (`not: { exists: ... }`) was inferred from the README.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
