# T5 self-report

- **Final artifact path:** `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single run succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - None encountered in practice, but the task's two decisive details were
    double-checked before writing: `fan_in_fan_out=true` means the
    `(B @ A)` product (Linear layout `[out, in]`) must be transposed to
    match the base Conv1D layout `[in, out]`; `scale = alpha / r = 2`.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard cap and must get its own
    shard; the greedy packer handles this as a special case explicitly.
- **Anything in the task text or documentation that was unclear:** No.
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3 only,
  used directly in a plain script rather than `peft.merge_and_unload()`.
  Reasoning: the merge is fully specified (name mapping, scale, transpose)
  and small (12 pairs), so a direct state-dict script keeps every
  correctness-relevant decision explicit and lets the required checks
  (pair count, no `lora_` names, shape, tensor count) run *before* any
  output is written, rather than being buried inside `transformers`
  model instantiation and PEFT's own merge path.
- **Approximate time spent, if you can tell:** ~10 minutes.
