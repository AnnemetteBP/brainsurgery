# Participant self-report: T5 (Pythia-1B, condition P)

- Final artifact path: `out/T5/solution.py` (output checkpoint in `out/T5/`: 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - PEFT adapter names carry the `base_model.model.` prefix that must be stripped to reach the base key, and `.lora_A.weight` / `.lora_B.weight` replaced by `.weight`.
  - Base weights are float16, so the merge is computed in float32 (`W.float() + scale * B @ A`) and cast back.
  - `adapter_config.json` lists `target_modules = ["query_key_value"]` while TASK.md says `attention.query_key_value`; the script derives targets from the adapter tensor names so this did not matter.
  - Greedy sharding in base key order needs to be checked against the 512 MiB tensor-data budget per shard; the two 206 MB embedding tensors fit inside a shard so no special-casing was needed.
- Anything in the task text or documentation that was unclear:
  - "A single tensor larger than that ... is stored alone in its own shard" is followed by examples (206 MB embeddings) that are not larger than the limit, so it is unclear whether the grader expects those two in their own shards or just enforces the per-shard budget. I applied the stated rule (per-shard budget only).
  - Shard file naming is unspecified; I used the HF convention `model-0000i-of-0000n.safetensors`.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 3 minutes
