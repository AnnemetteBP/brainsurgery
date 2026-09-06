# T5 report (condition F)

- Final artifact path: `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The task's parenthetical note that `model.embed_tokens.weight` and
    `lm_head.weight` ("412 MB each") are "larger than" the 512 MiB
    (536,870,912 B) shard budget is not numerically true (412,090,368 B <
    536,870,912 B); I followed the explicit numeric budget and the general
    "oversized tensor gets its own shard" rule literally rather than forcing
    these two tensors alone, and greedy bin-packing in base weight-map order
    happens to keep each of them close to solo anyway (shard 1 is
    `embed_tokens` alone, no other 393 MiB-class tensor shares a shard with
    another one).
  - `adapter_model.safetensors` keys use the PEFT wrapper prefix
    `base_model.model.model.layers.<i>...`, i.e. `model.` appears twice before
    `layers`; regex has to strip that whole prefix, not just `base_model.model.`.
- Anything in the task text or documentation that was unclear: the
  embed/lm_head shard-size claim noted above.
- Tools used (condition F): `safetensors` 0.5.3 (`safe_open`/`save_file` for
  chunked read and sharded write) and `torch` 2.14.0 (float32 matmul for
  `scale * B @ A` and tensor arithmetic). I did not use `peft.merge_and_unload`
  or `transformers.save_pretrained(max_shard_size=...)`: the merge formula is
  a single line given directly in the task (`scale = alpha/r`, `B @ A`, no
  transpose since `fan_in_fan_out=False`), and writing the shard packer myself
  gave exact, checkable control over the byte-exact 512 MiB budget and the
  "no adapter/intermediate tensor, no other tensor changed" requirements
  instead of trusting a black-box `max_shard_size` string parser.
- Approximate time spent, if you can tell: one write-run-verify cycle, no
  retries needed.
