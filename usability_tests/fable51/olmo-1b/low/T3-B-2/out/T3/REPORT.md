# Participant self-report: T3 (OLMo-1B-0724-hf), condition B

- Final artifact path: `out/T3/plan.yaml` (output checkpoint in `out/T3/`, 10 shards plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The `count` assert counts matches of a reference, not tensors of a dtype, so "exactly 112 bfloat16" was expressed as: the projection pattern counts to 112 and is all bfloat16, plus a negative-lookahead pattern asserting every other tensor is float32.
  - Regex references are full-match, so the projection pattern needed no anchors but the lookahead form needed a `$` inside the lookahead.
  - Sharded output is triggered by a directory-like `output.path` plus `shard: 256MB` (binary units, tensor data only), which matches the task's 256 MiB budget; oversized embeddings were placed alone in their own shards automatically.
- Anything in the task text or documentation that was unclear:
  - Whether an assert can count tensors by dtype directly; the docs suggest not, hence the two-pattern formulation.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 3 minutes
