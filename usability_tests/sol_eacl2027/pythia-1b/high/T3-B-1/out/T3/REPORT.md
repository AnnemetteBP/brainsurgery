# Participant self-report

- Final artifact path: `out/T3/model.safetensors.index.json` (with nine shard files in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None.
- Pitfalls or surprises you hit (one line each): The assertion language cannot directly count tensors by dtype, so the plan proves the exact bfloat16 count by asserting 64 projection-name matches are bfloat16 and every complementary tensor is float32.
- Anything in the task text or documentation that was unclear: The stated 206 MB embedding size appears to describe the float16 input; after the required float32 upcast each embedding is about 412 MB, but the documented oversized-single-tensor sharding rule handles this correctly.
- Tools used (condition F): Not applicable (condition B).
- Approximate time spent, if you can tell: 5 minutes.
