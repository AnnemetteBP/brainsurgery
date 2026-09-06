# T2 self-report (condition P)

- Final artifact path: `out/T2/solution.py` -> `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - Input is sharded, so tensors had to be gathered from both shards via `model.safetensors.index.json` before writing one flat file.
  - Slicing produces non-contiguous views that `safetensors` rejects or silently re-copies, so kept blocks are `.contiguous().clone()`d.
  - Row-block vs column-block asymmetry: q/k/v prune dim 0, o_proj prunes dim 1; matching on the `self_attn` path segment avoids touching the MLP `*_proj` tensors, which share the `proj` suffix.
  - The model is tied-embedding (no `lm_head.weight`), so 114 tensors = 16 layers x 7 + embedding + final norm.
- Anything in the task text or documentation that was unclear: nothing; the kept row/column ranges were given explicitly.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3 only).
- Approximate time spent, if you can tell: ~3 minutes.
