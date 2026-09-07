# T3 participant self-report (condition P)

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`:
  9 shards `model-0000N-of-00009.safetensors` + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution passed all checks.
- **Pitfalls or surprises you hit (one line each):**
  - The obvious `.*weight` / `endswith("weight")` targeting would have caught
    `gpt_neox.embed_in.weight`, `embed_out.weight` and every layer-norm weight,
    so I built the 64 projection names explicitly from `range(16)` x the four
    suffixes and asserted the set size instead of using a regex.
  - `gpt_neox.layers.<i>.attention.bias` is a uint8 causal-mask buffer, not a
    bias parameter — the projection biases that must stay (and be upcast to
    float32) are `attention.query_key_value.bias`, `attention.dense.bias`,
    `mlp.dense_*.bias`. Deleting by "anything named `*.bias`" would be wrong.
  - `masked_bias` is a 0-d scalar, so any size- or shape-based filter for
    buffers is fragile; I matched the three buffer suffixes by exact name.
  - Shard budget is on *output* bytes, not input bytes: the embeddings are
    206 MB as float16 in the input but 412 MB once upcast to float32, so both
    exceed the 256 MiB cap and must sit alone. I planned the shards from
    computed output byte counts (dtype size x numel) read from the safetensors
    header, before loading any tensor data.
  - Greedy packing puts an oversized tensor alone automatically only if the
    running shard is flushed first; I kept the "flush if non-empty and adding
    would overflow" order so the two embedding tensors land alone.
  - Two-pass structure (plan from header, then write shard by shard) also kept
    peak memory near one shard instead of the whole 2.4 GB state dict.
- **Anything in the task text or documentation that was unclear:**
  - The shard file naming scheme is not specified; I used the HuggingFace
    convention `model-00001-of-00009.safetensors`.
  - The index `metadata` block is not specified either; I wrote the customary
    `{"total_size": <sum of tensor bytes>}` alongside the required `weight_map`.
  - The iteration order that determines shard assignment is not specified; I
    used the input file's key order (safetensors returns keys sorted).
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3 only)
- **Approximate time spent, if you can tell:** ~3 minutes, dominated by the
  single ~90 s script run.
