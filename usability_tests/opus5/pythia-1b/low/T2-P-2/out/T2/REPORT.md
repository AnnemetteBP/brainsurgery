# T2 self-report (condition P)

- Final artifact path: `out/T2/model.safetensors` (script: `out/T2/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The GPT-NeoX fused QKV is interleaved per head (768-row blocks of q|k|v), not `[q | k | v]` segments, so pruning is a single contiguous block removal rather than three strided ones.
  - The output projection is `[out, in]`, so heads are *column* blocks of 256 there while they are row blocks in the QKV tensor.
  - `index_select` results need `.contiguous()` before `save_file` to avoid safetensors complaining about non-contiguous storage.
- Anything in the task text or documentation that was unclear: nothing; the layout and the exact index ranges to keep were spelled out.
- Tools used (condition F): n/a (condition P: torch + safetensors only).
- Approximate time spent, if you can tell: a few minutes.
