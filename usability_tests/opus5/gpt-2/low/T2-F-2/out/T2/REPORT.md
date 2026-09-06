# T2 self-report

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - Conv1D `[in, out]` layout means heads are *columns* of `c_attn.weight` but *rows* of `c_proj.weight`; slicing the same axis on both would silently produce a loadable but wrong checkpoint.
  - `c_attn` is three head-partitioned 768-wide segments, so the keep-index list must be built per segment and offset by 768, not applied once to the 2304-wide axis.
  - `attn.bias` (the causal mask buffer) shares the `attn.` prefix with real projections; a name pattern for head-bearing tensors would overmatch it and `attn.c_proj.bias`, so I named tensors explicitly and shape-asserted both untouched tensors.
  - `index_select` returns a fresh tensor but I call `.contiguous()` anyway, since safetensors rejects non-contiguous/aliased storage.
- Anything in the task text or documentation that was unclear: nothing; the task gave the exact keep-ranges, which I re-derived independently from head geometry and confirmed they agreed.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, plain script. I considered the suggested `transformers` `prune_heads` route but rejected it: it prunes a *live* `GPT2Model` and its `find_pruneable_heads_and_indices`/`prune_conv1d_layer` path re-emits tensors through a module round-trip, which risks dtype/layout drift and extra or renamed keys, whereas grading is bit-exact on a 160-key set. Direct `index_select` on the loaded state dict is exact by construction and shorter than configuring a tool.
- Approximate time spent, if you can tell: a few minutes; one execution, plus a separate independent verification pass comparing every tensor against the input.
