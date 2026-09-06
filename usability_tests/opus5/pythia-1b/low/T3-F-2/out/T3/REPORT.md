# T3 self-report

- **Final artifact path:** `out/T3/solution.py` (output: `out/T3/model-0000N-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none.
- **Pitfalls or surprises you hit:**
  - The over-broad-pattern hazard is real: I anchored the bfloat16 regex on the full
    tensor name (`^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$`)
    with escaped dots, so it cannot reach `embed_in.weight`, layer norms or biases.
  - `gpt_neox.layers.<i>.attention.bias` is a uint8 causal-mask buffer, not a projection
    bias — it must be dropped, while `attention.dense.bias` etc. must be kept and upcast.
    A `.*bias` rule would get this exactly backwards.
  - The two 206 MB embedding tensors exceed the 256 MiB shard budget only in aggregate,
    not alone; HF's splitter puts each in its own shard, which is what the spec asks for.
    My post-split assertion only flags an oversized shard when it holds more than one tensor.
- **Anything unclear:** the spec fixes the shard budget and the index contents but not the
  shard file naming; I used the HuggingFace convention (`model-0000i-of-0000N.safetensors`)
  on the assumption the hidden reference does the same.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — `load_file` / `save_file`. Direct state-dict access is the right
    granularity here: the task is per-tensor dtype policy plus key deletion, not model surgery.
  - `torch` 2.14.0 — `.to(torch.bfloat16)` / `.to(torch.float32)` for the casts.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards`, the same helper
    `transformers.save_pretrained` uses, so the shard layout and `weight_map` are the standard
    ones serving stacks expect rather than a hand-rolled bin-packing.
  - I deliberately did **not** route through `transformers` `save_pretrained(dtype=...)`: it
    applies one dtype to the whole model, so mixed precision would need post-hoc patching of
    its output, and it would re-materialize the buffers I am asked to drop.
- **Approximate time spent:** ~5 minutes.
