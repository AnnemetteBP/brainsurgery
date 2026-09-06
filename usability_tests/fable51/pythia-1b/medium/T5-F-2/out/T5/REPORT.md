# T5 self-report (condition F, Pythia-1B)

- Final artifact path: `out/T5/solution.py` (output checkpoint: `out/T5/model-0000N-of-00006.safetensors` + `out/T5/model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 passed all checks but packed the two embedding tensors together with other tensors (greedy fill, 4 shards); execution 2 stores them alone as the task text says (6 shards).
- Pitfalls or surprises you hit (one line each):
  - The task says a tensor "larger than 512 MiB" is stored alone and names the embeddings as examples, but they are 206 MB each, so a pure size rule would not isolate them; I added an explicit set for those two names.
  - PEFT prefix `base_model.model.` must be stripped and `.lora_A.weight` / `.lora_B.weight` replaced by `.weight` to find the base tensor.
- Anything in the task text or documentation that was unclear:
  - The sharding rule for the embeddings (see above). Whether shard file naming matters was not stated; I used the HF convention `model-XXXXX-of-XXXXX.safetensors`.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: read base and adapter, write shards (`safe_open`, `save_file`).
  - `torch` 2.14.0: float32 matmul `B @ A`, scaling, add, cast to float16.
  - Not used: `peft` `merge_and_unload` (would instantiate the model and give no control over the shard rule), `transformers` `save_pretrained` (its `max_shard_size` packing does not match the stated rule for the embeddings).
- Approximate time spent, if you can tell: about 5 minutes.

## Checks enforced by the script (fail with exit 1 before writing)
- exactly 16 complete A/B adapter pairs found and merged, each mapped to an existing base weight with matching delta shape;
- no output name contains `lora_`;
- `gpt_neox.layers.0.attention.query_key_value.weight` has shape [6144, 2048] and dtype float16;
- exactly 244 output tensors; every shard <= 512 MiB of tensor data; index maps all 244 names.

A separate one-off verification (not part of the artifact) confirmed 228 tensors bit-identical to the base and relative Frobenius error < 1e-3 on the 16 merged weights.
