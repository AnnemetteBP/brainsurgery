# T5 participant self-report (condition F, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/solution.py` (run as `python out/T5/solution.py` from the sandbox root). Output checkpoint: `out/T5/model-0000{1..10}-of-00010.safetensors` plus `out/T5/model.safetensors.index.json`.
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: did not fail (all checks passed), but the greedy packer put `model.layers.0.mlp.down_proj.weight` (67 MB) into the same shard as `model.embed_tokens.weight` (412 MB) since both fit under 512 MiB. TASK.md says the two 412 MB tensors are stored alone, so I deleted the output, added an explicit stand-alone set for those two names, and re-ran.
  - Execution 2: success (10 shards; 32 pairs merged; 114 tensors; layer-0 q_proj rel. err 3.6e-8).
- Pitfalls or surprises you hit (one line each):
  - The base checkpoint is not itself packed under the 512 MiB rule (shard 1 holds 113 tensors, ~4.7 GB), so the output had to be re-packed from scratch rather than mirroring the input sharding.
  - TASK.md says "a single tensor larger than [512 MiB]" is stored alone but then names two tensors of 412 MB, which are smaller than the budget; I followed the explicit statement and store them alone, in addition to the generic oversized rule.
  - `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj` (no `self_attn.` prefix) while TASK.md shows `self_attn.q_proj`; irrelevant to my solution since names are derived from the adapter tensor keys, not from `target_modules`.
  - Adapter keys carry the PEFT prefix `base_model.model.`; stripping it and replacing `.lora_{A,B}.weight` with `.weight` yields the base names.
- Anything in the task text or documentation that was unclear:
  - The stand-alone rule for the embedding and lm_head tensors (see above): the cap alone does not force them into their own shards, so the intended packing rule is under-specified. Also unclear whether the grader expects a specific shard count or file naming; I used the HF `model-XXXXX-of-YYYYY.safetensors` convention and greedy packing in the base's key order.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: float32 `B @ A`, scaling and addition.
  - `safetensors` 0.5.3: lazy reading of the base shards (`safe_open`/`get_slice` for the packing plan, `get_tensor` per shard) and `save_file` for output.
  - Python `json`/`re` (stdlib): index files and adapter-name mapping.
  - Not used: `peft`'s `merge_and_unload` and `transformers.save_pretrained`. They would instantiate the 5 GB model and give less control over packing (HF's greedy sharder would co-locate `embed_tokens` with the next tensor, contradicting TASK.md). A direct file-level script is also what the task describes.
- Approximate time spent, if you can tell: about 5 minutes wall clock, including one 9-second run per execution and a separate full verification pass (82 unchanged tensors bit-exact vs. base, 32 merged tensors worst relative Frobenius error 5.2e-8 vs. a float64 reference).

## Checks enforced by `solution.py`

Before writing: exactly 32 complete A/B pairs found; every adapter target exists in the base; base has 114 float32 tensors; no output name contains `lora_`; no destination shard file already exists; each planned shard is at most 536,870,912 bytes of tensor data. During/after writing: `B @ A` shape equals base shape; `model.layers.0.self_attn.q_proj.weight` is `[2048, 2048]` float32; 32 merges performed; 114 tensors written; written shards re-read and compared with the index `weight_map`; layer-0 q_proj checked against a float64 reference (rel. err < 1e-6). Any failure exits non-zero via `FAIL:`.
