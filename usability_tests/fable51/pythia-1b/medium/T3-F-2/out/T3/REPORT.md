# T3 participant self-report (condition F)

- Final artifact path: `out/T3/solution.py` (writes 9 shards + `model.safetensors.index.json` into `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Avoided the `.*weight` overmatch by anchoring the projection regex to the four exact module names under `gpt_neox.layers.<i>`.
  - Buffers are matched by an explicit anchored regex (`attention.bias|masked_bias|rotary_emb.inv_freq`) so no bias parameter is dropped; the script asserts exactly 48 drops.
  - Sharding rule is ambiguous about tensor order and shard naming; I used sorted key order, greedy first-fit packing, and HF-style `model-XXXXX-of-XXXXX.safetensors` names. Oversized embeddings land alone by construction.
- Anything in the task text or documentation that was unclear:
  - Whether the hidden reference expects a specific tensor ordering or shard file naming; the task only states the 256 MiB budget and the index format.
- Tools used (condition F): name, version, and why:
  - torch 2.14.0: dtype casts (`to(bfloat16)` / `to(float32)`) and bit-exact roundtrip comparison.
  - safetensors 0.5.3: `load_file` / `save_file` for input and shard output.
  - No transformers `save_pretrained`: it would not give per-tensor mixed precision, and loading the model would re-create the buffers I need to drop; a plain script gives exact control and easy checks.
- Approximate time spent, if you can tell: about 3 minutes.
