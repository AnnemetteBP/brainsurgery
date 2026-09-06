## Participant self-report

- Final artifact path: `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - None; the "at most 512 MiB tensor data per shard, one tensor alone if it
    exceeds that" rule needed care to satisfy exactly with a simple greedy
    bin-packer over tensors sorted by name.
- Anything in the task text or documentation that was unclear: none; the
  naming, scale (`alpha/r`), and `fan_in_fan_out` convention were fully
  specified.
- Tools used (condition F): `safetensors` 0.5.3 and `torch` 2.14.0 only, via
  a plain script — no `peft` `merge_and_unload`. Reasoning: `merge_and_unload`
  requires instantiating the full HF model (and PEFT wrapper) just to read
  off merged weights, which is unnecessary work and an unnecessary point of
  failure (dtype casts, device placement, module structure) when the task is
  a straightforward key-mapped matrix multiply-add on two safetensors files.
  A ~150-line script reading shards directly, merging by regex-derived
  layer/module index, and re-sharding with a simple greedy bin-packer gave
  full control over the output's exact shard boundaries and the required
  checks, with asserts that fail loudly (pair count, no `lora_` leakage,
  shape, tensor count) before any file is written.
- Approximate time spent, if you can tell: ~10 minutes.
