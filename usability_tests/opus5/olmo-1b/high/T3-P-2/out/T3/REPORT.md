# T3 self-report (condition P, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`:
  10 shards `model-0000N-of-00010.safetensors` + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):**
  - None. The single execution passed all pre-write and post-write checks.

- **Pitfalls or surprises you hit (one line each):**
  - Over-broad targeting was the obvious trap, so I never used a regex on `.*weight`;
    I built the 112 names explicitly as `model.layers.<i>.<suffix>` for the 7 listed
    suffixes and asserted each one exists with the shape the task states.
  - Sharding falls out exactly on the boundary: one layer is 128 MiB in bfloat16, so two
    layers total 268,435,456 bytes = exactly the 256 MiB budget. A greedy packer written
    with `>` (start a new shard only if the budget would be *exceeded*) fits 2 layers per
    shard; writing `>=` would have produced 16 half-full shards instead of 8 full ones.
  - `lm_head.weight` and `model.embed_tokens.weight` are 412 MB each, over the budget, so
    each is emitted alone. `tie_word_embeddings` is false in `config.json` and the two live
    in different input shards, so there was no shared-storage rejection from `save_file`.
  - The input is 5 GB in float32, so I did not build one big state dict: the plan is
    computed from safetensors headers (`get_slice().get_dtype()/get_shape()`, no data read)
    and then each output shard is loaded, cast and written on its own. Peak RSS stays at
    roughly one shard. Tensors are `.clone()`d before `save_file` because the `safe_open`
    mmap handle is closed by then.
  - Caught before running: in the post-write verification I had used the `safe_open` handle
    (`len(f.keys())`) after its `with` block had exited; I counted the tensors inside the
    block instead.

- **Anything in the task text or documentation that was unclear:**
  - The shard-file naming scheme is not specified. I used the HF convention
    `model-{i:05d}-of-{n:05d}.safetensors`, which is what the input checkpoint uses.
  - The tensor order used for packing is not specified, which determines which shard a
    tensor lands in. I used sorted key order; that is what the input `weight_map` and
    `safe_open().keys()` both give, so it is the least surprising choice.
  - "drop non-parameter buffers" appears in the objective, but the Input section says there
    are none here and requirement 3 says not to delete anything. I deleted nothing; the
    output key set is identical to the input's.
  - The index `metadata.total_size` is not required by the task. I wrote it (sum of tensor
    bytes, 2,971,664,384) to match the input index's shape.

- **Tools used (condition F):** n/a — condition P: torch 2.14.0+cu130, safetensors 0.5.3, stdlib only.

- **Approximate time spent, if you can tell:** ~5 minutes; the script itself runs in ~8 s wall clock.
