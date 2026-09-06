# T3 self-report

- Final artifact path: `out/T3/solution.py` (output: `out/T3/model-0000{1..4}-of-00004.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why: none.
- Pitfalls or surprises you hit:
  - `h.<i>.attn.bias` is the causal-mask buffer, not a bias parameter; the drop rule and the "keep biases in fp32" rule collide unless matched exactly (anchored regex, escaped dots).
  - A naive `.*weight` pattern would also hit `wte.weight`, `wpe.weight` and all `ln_*.weight`; I enumerated the four projection suffixes explicitly instead.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget, so the shard splitter must be allowed to emit it alone rather than erroring.
- Anything unclear: the exact shard-naming/packing convention is not spelled out; I assumed the HuggingFace convention (`model-XXXXX-of-YYYYY.safetensors`, greedy packing in state-dict order), which is what `huggingface_hub.split_torch_state_dict_into_shards` produces.
- Tools used (condition F):
  - `safetensors` 0.5.3 — load/save; the task's I/O format.
  - `torch` 2.14.0 — the `.to(torch.bfloat16)` cast (RNE) required by the spec.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards` for the canonical shard layout and index `weight_map`, so the sharding matches what serving stacks and the reference expect instead of a hand-rolled packer.
  - Not used: `transformers.save_pretrained` (its `dtype=` is uniform across the checkpoint, so it cannot express per-tensor mixed precision, and loading GPT-2 through a model class would rename/add keys); `mergekit` and `peft` are irrelevant to a pure dtype/shard export.
- Approximate time spent: ~5 minutes.
