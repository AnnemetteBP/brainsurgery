# T1 self-report (condition P, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/solution.py` (writes `out/T1/model.safetensors`, 86 tensors, float32)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Input is sharded (two files + index); loaded all shards into one dict via the index's `weight_map` and checked for duplicate keys across shards.
  - Renumbering collision hazard avoided by building a fresh dict from an explicit old->new map instead of renaming in place; an explicit collision check guards the write.
  - The "no tensor of blocks 12..15 remains" check is about the *output* index space, so it is implemented as "no output block index >= 12", separately from the drop of input blocks 2/6/10/14.
  - Called `.contiguous()` before `save_file` as a precaution; the loaded tensors were already contiguous and non-shared.
- Anything in the task text or documentation that was unclear: nothing material. The task lists the block map explicitly, which made the order-preserving renumbering straightforward.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3 only)
- Approximate time spent, if you can tell: a few minutes (one inspection, one script write + run, one read-back verification).
