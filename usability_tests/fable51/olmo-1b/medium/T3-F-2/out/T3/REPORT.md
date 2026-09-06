# T3 participant self-report (condition F, OLMo-1B-0724-hf)

- Final artifact path: `out/T3/solution.py` (output checkpoint in `out/T3/`: 10 shard files plus `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Avoided the over-broad `.*weight` pattern by anchoring the regex to `model.layers.<i>.{self_attn.{q,k,v,o}_proj | mlp.{gate,up,down}_proj}.weight`, so embeddings and lm_head are untouched.
  - Shard budget is on tensor data only, so the two 412 MB float32 embedding/lm_head tensors each get their own shard; the bf16 projections pack into 8 shards of exactly 256 MiB.
  - Used HF-style shard names and `{"format": "pt"}` metadata so transformers can load the directory.
- Anything in the task text or documentation that was unclear: the task does not fix a shard order or naming scheme; I used sorted key order with greedy fill and `model-XXXXX-of-XXXXX.safetensors` names.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: tensor loading and `to(torch.bfloat16)` (round-to-nearest-even).
  - `safetensors` 0.5.3: `safe_open` to read the input shards, `save_file` to write output shards.
  - Did not use `transformers.save_pretrained`, since its size-based sharding does not enforce a raw 256 MiB tensor-data budget and would have required instantiating the model; a direct script keeps dtype targeting explicit and the checks in one place.
- Approximate time spent, if you can tell: about 3 minutes.
