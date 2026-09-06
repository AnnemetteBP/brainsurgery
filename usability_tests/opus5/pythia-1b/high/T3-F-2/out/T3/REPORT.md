# T3 — Participant self-report

- **Final artifact path:** `out/T3/solution.py` (output: `out/T3/model-0000{1..9}-of-00009.safetensors` + `out/T3/model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none — the single execution succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious pattern route (`.*weight`) would hit `gpt_neox.embed_in.weight`, `embed_out.weight` and every layer-norm weight, so I built the 64 projection names explicitly from `range(16) x 4 suffixes` and asserted the set size instead of matching by regex.
  - `gpt_neox.layers.<i>.attention.bias` is a uint8 causal-mask buffer whose name looks exactly like a projection bias; only an explicit drop list separates it from the real `attention.dense.bias`, which must survive and become float32.
  - The task text quotes the embeddings as "206 MB each", which is their float16 input size; in the float32 output they are 412 MB, so they exceed the 256 MiB budget by a wider margin than the text suggests. Either way they land alone in their own shard.
  - The 256 MiB budget is per-shard *tensor data*, not file size — I measured `numel * element_size` rather than the on-disk size so the safetensors header does not count against the budget.
  - I did not use `transformers.save_pretrained` even though `F-allowed.md` suggests it as the T3 route: it applies a single `dtype` to the whole model, so it cannot express a per-tensor mixed precision. I used only its sharding half, via the `huggingface_hub` splitter it calls internally.

- **Anything in the task text or documentation that was unclear:**
  - The shard *file naming* is not specified, only that there are "shard files plus an index file `model.safetensors.index.json`". I assumed the canonical HuggingFace `model-{n}-of-{total}.safetensors` pattern, since that is what serving stacks and the index format imply.
  - Whether the index should carry a `metadata.total_size` field is unspecified; I included it, as HF's own index does.
  - Whether `config.json`/tokenizer files should be copied into `out/T3/` is unspecified. "Required result" lists only shard files and the index, so I wrote nothing else.
  - The per-shard packing order is unspecified, so the exact tensor-to-shard assignment depends on state-dict iteration order; I used the safetensors key order unchanged and relied on the HF splitter.

- **Tools used (condition F):**
  - `torch` 2.14.0 — dtype casts (`.to(torch.bfloat16)` / `.to(torch.float32)`) and dtype assertions.
  - `safetensors` 0.5.3 — `load_file` / `save_file` for reading the input and writing each shard.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards`, the same splitter `transformers.save_pretrained` uses, so the shard layout and `weight_map` are the canonical HF ones rather than something I hand-rolled. It also implements the "a single tensor bigger than the budget gets its own shard" rule the task asks for.
  - Deliberately *not* used: `transformers.save_pretrained` (single global dtype, cannot express mixed precision), `mergekit` (merge-oriented; its dtype conversion is also model-wide), `peft`, `torch-state-bridge` (this task renames nothing).
  - Rationale overall: the task is per-tensor dtype selection plus a drop list — a 90-line script expresses that exactly, and the only part worth delegating is the sharding, where matching the established HF layout matters more than my own packing.

- **Approximate time spent, if you can tell:** ~10 minutes, most of it inspecting the input header and deciding the sharding route.

## Required checks, as enforced

All four run in `solution.py` *before* any file is written; each raises `SystemExit("CHECK FAILED: ...")`:

1. exactly 64 bfloat16 tensors;
2. `gpt_neox.layers.0.attention.query_key_value.weight` is bfloat16;
3. `gpt_neox.embed_in.weight` is float32;
4. exactly 196 output tensors.

Plus, as extra guards: every targeted name exists in the input (no silent no-match), the four projection shapes are as specified, the non-float32 set is *equal* to the 64 projections (catches over-match), the dropped key set is *equal* to the 48 buffers (catches dropping a parameter), every shard's tensor data is within 256 MiB unless it is a single oversized tensor, the shards together hold 196 tensors, and `weight_map` covers every output tensor.
