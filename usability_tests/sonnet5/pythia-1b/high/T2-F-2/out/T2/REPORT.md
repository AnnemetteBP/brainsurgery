# T2 report (condition F, pythia-1b)

## Participant self-report

- Final artifact path: `out/T2/solution.py` (invoked via `out/T2/run.sh`), output at `out/T2/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None encountered in practice, but two things needed care up front: the QKV block to drop is 768 rows (interleaved q/k/v per head), not a 256-row slice from a `[q|k|v]`-segmented layout, and it must be removed from the *middle* of the row range (rows 3840..4607), not from one end.
  - `dense.weight` heads are column blocks (`[:, h*256:(h+1)*256]`), the transpose of the row-block layout in `query_key_value`, since `dense` is a `[2048, 2048]` `nn.Linear` consuming the concatenated head outputs as input features.
- Anything in the task text or documentation that was unclear: no, the task text fully specified the row/column ranges to keep, which made verification by direct index arithmetic straightforward.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0 and `safetensors` 0.5.3 only, for tensor loading, slicing, and saving. No merge-config tool (`mergekit`) or `transformers.prune_heads` was used: `prune_heads` is written for encoder self-attention modules with separate q/k/v projections and doesn't address GPTNeoX's fused, interleaved `query_key_value` tensor, and `mergekit`'s recipes (passthrough layer slicing, task arithmetic) operate on whole tensors/layers, not sub-tensor head slices. The task's row/column boundaries are exact and fully specified in `TASK.md`, so a short, directly auditable slicing script was the smallest correct tool, and every required check runs as a plain `assert` against measured shapes before the file is written.
- Approximate time spent, if you can tell: a few minutes of reasoning about the layout plus one script write and one execution.
