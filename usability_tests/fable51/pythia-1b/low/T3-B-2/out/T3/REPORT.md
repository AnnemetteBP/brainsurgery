# Participant self-report: T3 (Pythia-1B), condition B

- Final artifact path: `out/T3/plan.yaml` (output: `out/T3/model-0000{1..9}-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all asserts and wrote 196 tensors across 9 shards.
- Pitfalls or surprises you hit (one line each):
  - "Exactly 64 tensors are bfloat16" cannot be expressed as a dtype-based count, so it is encoded as: the projection pattern matches exactly 64 tensors, all of them are bfloat16, and every tensor outside that pattern (negative-lookahead regex) is float32.
  - The plan casts everything to float32 first, then the 64 projection matrices to bfloat16 (float16 -> float32 -> bfloat16 is exact in the first step, RNE in the second, same as a direct cast).
  - Shard sizes are binary units, so `256MB` in the plan equals the required 268,435,456 bytes; the two 412 MB embedding tensors ended up alone in their own shards as expected.
- Anything in the task text or documentation that was unclear: the README does not state explicitly whether `dtype` with a multi-match reference checks every match, but the run confirmed it does (the projection pattern and the negative-lookahead pattern both passed and the header counts are 64 BF16 / 132 F32).
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
