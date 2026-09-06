## Participant self-report

- Final artifact path: `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution
  succeeded.
- Pitfalls or surprises you hit (one line each):
  - none of real consequence; `fan_in_fan_out=false` and the `[out, in]`
    layout match for both the adapter factors and the base weight, so no
    transpose was needed — confirmed this against `adapter_config.json`
    before merging rather than assuming it.
  - had to derive the base tensor name from the adapter tensor name by
    stripping the `base_model.model.` prefix and the `.lora_A`/`.lora_B.weight`
    suffix; used a regex instead of string slicing to make the mapping
    explicit and to fail loudly on any adapter tensor name it doesn't
    recognize.
- Anything in the task text or documentation that was unclear: no — inputs,
  the merge formula, and the sharding budget were all fully specified.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, in a
  plain script. Skipped `peft.merge_and_unload`: it requires instantiating
  the full HF `GPTNeoXForCausalLM` (extra memory and dependency surface) to
  do a rank-16 matmul-and-add over 32 known tensors whose naming convention
  is already given in `adapter_config.json`; direct safetensors read/merge/
  write is simpler and gives direct control over the required per-shard
  512 MiB tensor-data budget (with oversized tensors — the two 206 MB
  embedding tensors — alone in their own shard), which the task requires
  the checkpoint to enforce.
- Approximate time spent, if you can tell: a few minutes of thinking/writing,
  well under a minute to run.
