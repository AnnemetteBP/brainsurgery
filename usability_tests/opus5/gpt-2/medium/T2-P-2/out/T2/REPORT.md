# T2 participant self-report

- Final artifact path: `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - GPT-2's Conv1D `[in, out]` layout means the head axis is dim 1 for `c_attn` but dim 0 for `c_proj`; slicing the same axis on both would have been silently wrong.
  - `c_attn` is fused `[q | k | v]`, so the head-5 block must be dropped three times, once per 768-wide segment, with the segment order preserved.
  - `index_select` output was made `.contiguous()` before saving, since safetensors rejects non-contiguous / storage-sharing tensors.
  - `attn.bias` (the causal mask buffer) sits in the attention namespace but is not head-bearing; it had to be left alone rather than caught by a name pattern.
- Anything in the task text or documentation that was unclear: nothing; the explicit kept-index ranges made the layout unambiguous and let me verify my computed indices against the spec.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~3 minutes
