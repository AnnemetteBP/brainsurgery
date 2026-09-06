# Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The head-bearing axis for `dense.weight` is the *input* (column) axis,
    not the output/row axis, since it's an `nn.Linear [out, in]` consuming
    the concatenated per-head attention outputs — easy to slice the wrong
    axis if you assume "head-bearing" always means rows.
  - The interleaved GPT-NeoX qkv layout (per-head 768-row blocks, each
    block itself `[q(256)|k(256)|v(256)]`) means pruning a head is a single
    contiguous 768-row deletion, not three separate 256-row deletions from
    global q/k/v segments — worth double-checking against the task's
    explicit row ranges before trusting the "interleaved" description.
- Anything in the task text or documentation that was unclear: none; the
  explicit kept-row/kept-column ranges in "Required result" made it easy to
  cross-check the index arithmetic.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes
