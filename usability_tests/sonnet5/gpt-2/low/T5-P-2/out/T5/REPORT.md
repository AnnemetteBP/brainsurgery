## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Had to remember the Conv1D `[in, out]` vs Linear `[out, in]` layout, so `(B @ A)` needed a transpose before adding to the base weight (`fan_in_fan_out=True` signals this).
  - Adapter tensor names use a `base_model.model.` prefix that must be stripped to map to base checkpoint keys.
  - Sizing shards required computing raw tensor bytes (numel * element_size) against the 100 MiB cap and special-casing `wte.weight`, which alone exceeds the cap.
- Anything in the task text or documentation that was unclear: none; the spec gave the exact formula, scale, and shard rule.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes to write and verify the script.
