## Participant self-report

- Final artifact path: `out/T3/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `cast_` (in-place, same name) is the right tool here, not `cast`, since tensor
    names must not change; `cast` creates a new destination tensor.
  - The 48 target projection matrices needed an explicit alternation regex
    (`attn.(c_attn|c_proj)|mlp.(c_fc|c_proj)`) rather than a broad `.*weight`,
    which would also match embeddings, layer norms and biases.
  - `assert.count`/`assert.dtype` take a tensor-reference pattern in `of`, not a
    dtype predicate, so "exactly 48 bfloat16 tensors" had to be checked by
    asserting the count and dtype of the same explicit projection-matrix
    pattern used for casting, rather than a dtype-based filter.
  - `output.shard: 64MB` combined with a directory output path was enough to
    get sharding with an index file; verified afterward that the >64MiB
    `wte.weight` correctly lands alone in its own shard per the documented
    oversize-tensor rule.
- Anything in the task text or documentation that was unclear: none; the
  README's notes on shard budgets (tensor-data-only, oversized tensor gets
  its own shard) and the `equal`/`dtype`/`count` assert semantics were
  sufficient to write and verify the plan without trial and error.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: under 10 minutes.
