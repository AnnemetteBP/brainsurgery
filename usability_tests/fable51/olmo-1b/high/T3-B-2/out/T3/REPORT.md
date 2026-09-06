# Participant self-report: T3 (OLMo-1B-0724-hf), condition B

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`: `model-0000N-of-00010.safetensors` x10 plus `model.safetensors.index.json`; `out/T3/summary.yaml` is the executed-plan summary written via `--summarize-path`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The README warns that `.*weight` would also hit `model.embed_tokens.weight` and `lm_head.weight`, so the `cast_` target is an explicit regex over `self_attn.[qkvo]_proj` and `mlp.(gate|up|down)_proj` only, and a `count` assert pins it to 112 matches before casting.
  - No assert operator counts tensors by dtype directly, so "exactly 112 tensors are bfloat16" is expressed as: the projection pattern matches 112 and is bfloat16, the complement pattern (negative lookahead) matches 2 and is float32, and `.*` matches 114.
  - `output.shard` uses binary units, so `256MB` is exactly 268,435,456 bytes; the resulting bf16 shards each carry exactly that many bytes of tensor data (14 x 16 MiB / 32 MiB tensors), and the two 412 MB float32 tensors landed alone in shards 1 and 2 as required.
  - The plan file lives inside the output directory (`out/T3/plan.yaml`) as the rules require; brainsurgery wrote its shards next to it without complaint.
- Anything in the task text or documentation that was unclear:
  - `assert.dtype` help does not say whether `of` may match several tensors and whether every match is checked; the run accepted a multi-match pattern, and shell inspection of the safetensors headers confirmed all 112 projections are BF16 and both other tensors F32, but the docs could state the multi-match semantics explicitly.
- Tools used (condition F): n/a (condition B). Post-run verification used `jq` and `od` on the index and safetensors headers, no Python.
- Approximate time spent, if you can tell: about 3 minutes (reading the docpack, one plan run, header verification).
