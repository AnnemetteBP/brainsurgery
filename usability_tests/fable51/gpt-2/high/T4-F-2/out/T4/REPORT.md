# T4 self-report (condition F, GPT-2 124M)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 succeeded.
- Pitfalls or surprises you hit (one line each):
  - The checkpoints use prefix-less GPT-2 keys (`h.<i>...`, `wte`, no `transformer.`) and include the 12 `h.<i>.attn.bias` causal-mask buffers; mergekit's architecture-based loader would likely drop or rename those, breaking the exact 160-key / bit-exact requirement, so I did not route through it.
  - `ft1`/`ft2` safetensors files carry no `format: pt` metadata while `base` does; I wrote the output with `{"format": "pt"}` like the base.
- Anything in the task text or documentation that was unclear: nothing material. The ordering hazard (task vectors against the unmodified base) is handled by reading all three dicts before computing anything and never mutating `base`.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file` / `save_file` for reading the three checkpoints and writing the single-file output.
  - `torch` 2.14.0: `torch.equal` for the bit-exact shared-tensor verification and float32 arithmetic for the merge.
  - Not used: `mergekit` 0.1.4 task-arithmetic. It cannot express the precondition check (non-MLP tensors identical across all three), the "exactly 48 merged" check, or guarantee the raw key layout with `attn.bias` buffers, so a short script was safer than a YAML config plus a wrapper to re-verify its output.
- Approximate time spent, if you can tell: a few minutes; one inspection of the key layout, one script, one run.

## Checks enforced by the script (all abort with a non-zero exit)

1. Same tensor-name set in base/ft1/ft2, exactly 160 names, all float32, identical shape/dtype per name.
2. The 48 MLP names are built explicitly (12 layers x c_fc/c_proj x weight/bias) and cross-checked against a regex; every non-MLP tensor is `torch.equal` in base vs ft1 and base vs ft2 before any merge happens.
3. Exactly 48 tensors merged; output dict has exactly 160 keys equal to the base key set.
4. After writing, the file is re-loaded and verified: 160 keys, shapes/dtypes match, non-MLP tensors bit-exact to base, MLP tensors within 1e-6 relative Frobenius error of the formula.
