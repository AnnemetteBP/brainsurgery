# T5 (OLMo-1B-0724-hf), condition B: participant self-report

- Final artifact path: `out/T5/plan.yaml` (executed summary in `out/T5/summary.yaml`; output checkpoint in `out/T5/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Output alias inference: with two inputs the plan must write to exactly one alias, so the `B @ A` intermediate had to be created on the `base` alias (temporary `merge_delta.*` names) rather than on `lora`, and deleted from `base` afterwards; a `matmul` destination on `lora` would have made the output ambiguous.
  - `matmul`/`add_` pair the A/B factors and the base weight through regex capture rewrite (`\1`, `\2`) from `from_a`, which is documented only in a short "Mapping note" in the interfaces reference.
  - `lm_head.weight` and `model.embed_tokens.weight` (412,090,368 bytes each) are smaller than the 512 MiB budget, so by the documented packing rule (state-dict order, fill up to the budget) `lm_head` ends up alone in shard 1 and `model.embed_tokens` shares shard 2 with `model.layers.0.mlp.down_proj.weight`; every shard stays within 536,870,912 bytes of tensor data.
- Anything in the task text or documentation that was unclear:
  - TASK.md says the two 412 MB tensors are "larger than" the 512 MiB budget and are stored alone; they are not larger, so I followed the tool's documented packing rule and the stated per-shard limit instead of forcing them into single-tensor shards.
  - The scale factor was applied as `scale_` by 2.0 on the delta before `add_` (exact in float32); the task does not say whether `scale * (B @ A)` or `(scale * B) @ A` is the reference, but both are within the 1e-5 tolerance.
- Tools used (condition F): n/a (condition B). Post-run verification of the written files was done with a throwaway Python read-only check, not part of the solution.
- Approximate time spent, if you can tell: about 5 minutes (docs reading, one plan run of ~10 s, verification).
