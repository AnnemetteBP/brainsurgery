## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the plan ran and passed all assertions on the first execution.
- Pitfalls or surprises you hit (one line each):
  - Intermediate `matmul`/`scale` outputs and the in-place `add_` had to land on the same alias (`base`) as everything else, since `output` requires all writing transforms to resolve to a single alias; using the `lora` alias for the scratch tensors would have made the output alias ambiguous.
  - `matmul`/`scale` destinations must not already exist and must not be sliced, but they do accept capture-group interpolation (`\1`, `\2`) the same way `copy` does, so all 32 layer/module pairs could be merged in one `matmul` + one `scale` + one `add_` block instead of 32 separate blocks.
  - The doc pack's note that `model.embed_tokens.weight` and `lm_head.weight` (412 MB each) "are stored alone in their own shard" doesn't hold under the tool's documented greedy, state-dict-order packing at a 512 MiB budget: neither tensor individually exceeds the budget, so the packer is free to combine one of them with a smaller tensor if it fits. In this run `lm_head.weight` ended up alone (nothing before it, and adding `model.embed_tokens.weight` next would have overflowed) but `model.embed_tokens.weight` was packed together with `model.layers.0.mlp.down_proj.weight` (479 MiB total, still under budget). I left this as the tool's default, documented behavior rather than trying to force a specific shard layout, since nothing in the "Required checks" section calls for it and the per-shard budget and tensor/key-set correctness all check out.
- Anything in the task text or documentation that was unclear: the "Required result" parenthetical about `model.embed_tokens.weight`/`lm_head.weight` being individually forced into their own shard is stated as fact but isn't implied by the documented packing rule ("a single tensor larger than the budget is written alone") since both tensors are under the 512 MiB budget — see pitfall above.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~15 minutes (reading docs/examples, writing the plan, verifying output).
