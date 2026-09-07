# Participant self-report — T2 (condition F)

- **Final artifact path:** `out/T2/solution.py` (invoked via `out/T2/run.sh`), output at `out/T2/model.safetensors`.
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single run succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - `transformers==5.12.1`'s GPT-2 implementation has no `prune_heads` on `GPT2Attention`/`GPT2Model` (only `pytorch_utils.prune_linear_layer`, which doesn't handle `Conv1D`), so the API route suggested for T2 wasn't available; fell back to a direct safetensors/torch script.
  - Confirmed up front that `c_attn` is `[q(768) | k(768) | v(768)]` columns and `c_proj` heads are row blocks (Conv1D `[in, out]` layout means "columns" of `c_attn` and "rows" of `c_proj` are both the head axis, not a transposed one) before writing the mask, to avoid slicing the wrong axis.
- **Anything in the task text or documentation that was unclear:** No — the exact column/row ranges given in the spec matched the derivation from a plain per-head boolean mask (verified by an explicit assertion in the script).
- **Tools used (condition F):** `torch` 2.14.0 and `safetensors` 0.5.3 only — a plain script over tensors loaded/saved with `safetensors.torch`. Did not use `transformers` (no working `prune_heads` in this version), `mergekit` (this is per-head slicing inside single tensors with two axes, not layer-level passthrough splicing), `peft` (no adapters involved) or `torch-state-bridge` (no key renaming needed, only value slicing).
- **Approximate time spent, if you can tell:** ~5 minutes (checkpoint inspection, script, one run, verification).
