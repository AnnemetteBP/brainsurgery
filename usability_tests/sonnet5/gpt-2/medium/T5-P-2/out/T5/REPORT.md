## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Had to keep in mind that base `h.<i>.attn.c_attn.weight` is Conv1D `[in, out]` while `lora_A`/`lora_B` follow the `nn.Linear` convention, so the merge is `scale * (B @ A).T`, not `B @ A` directly (`fan_in_fan_out=true` is the signal for this).
  - Sharding required bin-packing by tensor-data bytes only (header bytes don't count toward the 100 MiB budget), and the single oversized tensor (`wte.weight`, ~154 MB) needed to go alone in its own shard rather than failing the size check.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~10 minutes.
