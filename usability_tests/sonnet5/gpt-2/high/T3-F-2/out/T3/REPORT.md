# T3 participant self-report (condition F)

- **Final artifact path:** `out/T3/solution.py` (invoked via `out/T3/run.sh`),
  writing to `out/T3/model-000{01..04}-of-00004.safetensors` and
  `out/T3/model.safetensors.index.json`.
- **Number of times you executed the script or plan:** 1 (the one clean run
  that produced the committed output). I separately exercised the assertion
  logic in a throwaway interactive check (deliberately shrinking the bf16
  target set to confirm the count check fires before any file is written);
  that was not a run of the solution itself and left no output, so it isn't
  counted as an attempt.
- **Which executions failed, and why:** none. The single execution of
  `solution.py` succeeded on the first try.
- **Pitfalls or surprises you hit:**
  - `h.<i>.attn.bias` is the causal-mask buffer, not a parameter — has to be
    excluded from both the cast and the "unchanged" check, and dropped
    entirely from the output.
  - Sorting keys lexicographically for a deterministic shard order would put
    `h.10.*`/`h.11.*` between `h.1.*` and `h.2.*`; used an explicit numeric
    traversal (`wte`, `wpe`, then layers `0..11` in a fixed sub-key order,
    then `ln_f`) instead of `sorted(state_dict.keys())`.
  - `wte.weight` (≈147 MiB) is larger than the 64 MiB shard budget by
    itself, so the greedy packer needs an explicit "oversized tensor gets
    its own shard" branch rather than just erroring when a lone tensor
    already exceeds the budget.
  - Shard budget is tensor-data bytes only, not file size on disk (safetensors
    headers add a small amount); computed budgets from `numel() * element_size()`
    rather than from the written file size.
- **Anything in the task text or documentation that was unclear:** No —
  the shapes, key names, and exact expected counts (48 bf16, 148 total) in
  TASK.md were enough to write direct assertions against.
- **Tools used (condition F): name, version, and why:**
  - `torch` 2.14.0 — `.to(torch.bfloat16)` for the round-to-nearest-even
    cast, and to hold/compare tensors.
  - `safetensors` 0.5.3 (`safetensors.torch.load_file`/`save_file`) —
    loading the input checkpoint and writing shards directly.
  - No merge/adapter framework (`mergekit`, `peft`, `torch-state-bridge`,
    `transformers` sharded save) was needed: this task is a per-tensor dtype
    cast, a buffer drop, and a byte-budgeted repack of a single checkpoint,
    which `torch` + `safetensors` express directly and let the required
    checks run as plain Python asserts before anything is written. A
    higher-level tool would have added indirection (e.g. discovering how to
    express "cast only these keys" through a merge config or an HF
    `torch_dtype=`/component-map) without simplifying the actual logic.
- **Approximate time spent, if you can tell:** Not tracked precisely; the
  script was written and validated in a single pass with no failed
  executions or backtracking.
