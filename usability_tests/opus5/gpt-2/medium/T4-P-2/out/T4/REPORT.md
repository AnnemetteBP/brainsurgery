# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The MLP key set has to be constructed explicitly (12 layers x {c_fc,c_proj} x {weight,bias} = 48) rather than matched by a regex on `mlp.` — a loose pattern would be easy to overreach with, and the explicit list doubles as the "exactly 48" check.
  - The ordering hazard is only avoided by keeping `base`, `ft1`, `ft2` as three separate dicts and writing into a fresh `out` dict; mutating `base` in place while merging would make the second task vector be taken against an already-modified base.
  - `torch.equal` on the 112 non-MLP tensors is the bit-exact comparison the spec wants; an `allclose` would silently accept a drifted backbone.
  - Tensors are cloned and made contiguous before `save_file`, since safetensors rejects tensors sharing storage.
- Anything in the task text or documentation that was unclear:
  - The task says the merge is computed in float32 and states the inputs are already float32, so the `.to(torch.float32)` casts are defensive no-ops; it was not stated whether the output dtype should follow the base or be forced to float32 (I preserved the base dtype, which is the same thing here).
  - "Verify the three checkpoints have the same tensor names" does not say whether shapes/dtypes must also match; I checked those too for the MLP tensors and implicitly via `torch.equal` for the rest.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: ~5 minutes.
