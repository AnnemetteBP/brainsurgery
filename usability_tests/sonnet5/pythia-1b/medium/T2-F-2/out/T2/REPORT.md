# T2 report (condition F)

- Final artifact path: `out/T2/solution.py` (invoked via `out/T2/run.sh`),
  output at `out/T2/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run
  succeeded.
- Pitfalls or surprises you hit (one line each):
  - None of real consequence — the spec's row/column ranges for the fused
    QKV block (768 rows/head) and the dense input columns (256 cols/head)
    were given explicitly enough that a direct slice-and-checkpoint script
    was safer than reaching for `transformers.prune_heads`, since GPT-NeoX's
    interleaved per-head QKV layout isn't guaranteed to match whatever
    generic pruning helper `prune_heads` uses internally, and TASK.md gave
    exact row/column boundaries to verify against instead.
- Anything in the task text or documentation that was unclear: no; the row
  ranges (`0..3839`, `4608..6143`) and column ranges (`0..1279`, `1536..2047`)
  were given directly, and matched a plain block-drop of head 5's 768-row /
  256-column slice.
- Tools used (condition F): `torch` 2.14.0 (tensor indexing) and
  `safetensors` 0.5.3 (`load_file`/`save_file`) — a plain script on top of
  these two was simpler and more directly checkable against the spec's exact
  block boundaries than routing through `transformers.prune_heads`,
  `mergekit`, or `torch-state-bridge`, none of which have first-class support
  for GPT-NeoX's interleaved fused-QKV head layout.
- Approximate time spent, if you can tell: a few minutes (single-pass script,
  first execution passed all checks).
