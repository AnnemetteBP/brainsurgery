# Participant self-report: T1 (GPT-2 124M, condition P)

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard avoided by building a fresh output dict from an explicit old-to-new index map instead of renaming in place, with an explicit collision check on every insert.
  - The regex anchors on `^h\.(\d+)\.` so `attn.bias` mask buffers and `mlp.c_proj` are carried by block index, never by suffix matching.
  - `save_file` needs contiguous tensors; called `.contiguous()` defensively (a no-op here).
- Anything in the task text or documentation that was unclear: the "Required checks" section says "no tensor of blocks 9, 10, 11 remains"; I read this as "no block index >= 9 in the output" and implemented it that way.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes.
