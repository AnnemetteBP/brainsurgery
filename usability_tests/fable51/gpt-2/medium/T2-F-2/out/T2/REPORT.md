# T2 report (condition F, GPT-2 124M)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Stock HF `GPT2Config` refuses `n_head=11` with `n_embd=768` (divisibility check), so the pruned checkpoint cannot be loaded as an 11-head model through `transformers` without a custom attention class or `pruned_heads`; this only affected an optional post-hoc sanity load, not the deliverable.
  - `attn.bias` is the causal mask buffer, not a per-head tensor; it is matched by name suffix so it passes through untouched.
- Anything in the task text or documentation that was unclear: nothing; the explicit column/row ranges made the layout unambiguous.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load and save the checkpoint (bit-exact, dtype-preserving).
  - `torch` 2.14.0: `index_select` on the head-block indices; `torch.equal` for checks.
  - I did not use `transformers.prune_heads` because it keeps the model at 12 heads with a `pruned_heads` config entry and writes HF-prefixed keys; a direct slice gives the required 11-head layout with unchanged names and bit-exact values.
- Approximate time spent, if you can tell: about 3 minutes.

## Checks enforced in `solution.py` (all before writing)

- `h.0.attn.c_attn.weight` is `[768, 2112]`, `h.0.attn.c_attn.bias` is `[2112]`, `h.0.attn.c_proj.weight` is `[704, 768]`, output has exactly 160 tensors.
- Additionally: same shapes on all 12 layers, key set and dtypes unchanged, every non-head-bearing tensor bit-identical to the input, block-boundary spot check (retained block 5 equals source head 6), and a post-write reload check.
