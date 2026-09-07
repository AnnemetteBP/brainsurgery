# Participant self-report: T5 (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2 (the first run's output was truncated by my log filter, so I re-ran it to read the log; both runs succeeded)
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Output alias inference with two inputs: every write (matmul destination, scale_, add_, delete) had to be kept on the `base` alias, so the deltas were created as `base::<module>.lora_delta` and deleted after the add.
  - `matmul` pairs `from_b` and `to` as regex rewrites of `from_a`, so the capture group from the lora_B name is reused to locate lora_A and the base weight.
- Anything in the task text or documentation that was unclear: `matmul`'s help does not state how `from_b` is paired with `from_a` for multi-match references; inferred from the `add` example and the `equal` assert docs.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
