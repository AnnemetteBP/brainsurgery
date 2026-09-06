# T5 participant self-report (condition F, Pythia-1B)

- Final artifact path: `out/T5/solution.py` (output checkpoint: `out/T5/model-0000N-of-00006.safetensors` + `out/T5/model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 produced a valid 4-shard checkpoint (all shards under 512 MiB, all checks passed) but packed `embed_in` together with layer tensors; the task says the two 206 MB embeddings must be stored alone, so I added an explicit "store alone" rule (tensors >= 128 MiB) and re-ran, giving 6 shards.
- Pitfalls or surprises you hit (one line each):
  - The task's sharding clause says the embeddings are "larger than" the 512 MiB budget, which they are not (206 MB each); I followed the stated intent (alone in their own shard) rather than the literal rule.
  - `adapter_config.json` lists `target_modules = ["query_key_value"]`, not `attention.query_key_value` as TASK.md says; irrelevant since I mapped names from the adapter keys by stripping the `base_model.model.` prefix and `.lora_{A,B}.weight` suffix.
- Anything in the task text or documentation that was unclear: the embedding-alone sharding rule above (contradicts the numeric budget).
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: float32 matmul `scale * B @ A`, add, cast to float16.
  - `safetensors` 0.5.3: `safe_open` to load base and adapter, `save_file` per shard.
  - Plain script instead of `peft.merge_and_unload`: avoids instantiating the model, keeps the 228 unchanged tensors bit-exact, and gives direct control of shard packing and the required checks. Verified afterwards with a separate torch script: 228 tensors bit-identical to base, 16 `query_key_value.weight` changed, all shapes/dtypes preserved.
- Approximate time spent, if you can tell: about 3 minutes.
