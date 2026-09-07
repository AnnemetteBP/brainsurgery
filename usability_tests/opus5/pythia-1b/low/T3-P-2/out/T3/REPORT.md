# T3 self-report (condition P)

- Final artifact path: `out/T3/solution.py` (output in `out/T3/`, 9 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Anchored the projection regex with `^...$` and escaped dots so it cannot reach
    `embed_in.weight`, `embed_out.weight`, layer norms or the `.bias` entries.
  - `gpt_neox.layers.<i>.attention.bias` is a uint8 mask buffer, not a projection bias;
    it is matched by the buffer-deletion pattern, while `...query_key_value.bias` is a
    parameter and is kept (upcast to float32).
  - Greedy shard packing needs the "flush before adding" order so a >256 MiB tensor ends
    up alone; both 206 MB embedding matrices did (shards 1 and 2).
  - Preserved the original safetensors header order (read from the raw header) rather than
    `keys()` sorted order, so the shard layout follows the checkpoint's own tensor order.
  - Saved with `.contiguous()` to avoid safetensors rejecting shared/strided storage.
- Anything in the task text or documentation that was unclear:
  - The shard file naming convention and the index `metadata.total_size` field are not
    specified; I used the HuggingFace convention `model-{i:05d}-of-{n:05d}.safetensors`.
  - The tensor ordering used for packing shards is not specified either.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes.
