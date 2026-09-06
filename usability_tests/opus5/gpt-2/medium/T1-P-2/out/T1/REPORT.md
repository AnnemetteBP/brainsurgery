# T1 participant self-report (condition P)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The renumbering collision hazard is real but disappears if you build a fresh output dict instead of renaming in place; I still added an explicit collision check on the destination name.
  - `h.<i>.attn.bias` is a causal-mask buffer, not a projection bias, so a name-based filter must treat it as an ordinary block tensor (13 per block, not 12).
  - The block regex has to be anchored (`^h\.(\d+)\.`) so `mlp.c_proj` / `attn.c_proj` suffixes are never touched and only the leading index is rewritten.
  - Kept the source file's safetensors metadata and called `.contiguous()` before saving to avoid shared/non-contiguous storage rejections.
- Anything in the task text or documentation that was unclear:
  - The check "no tensor of blocks 9, 10, 11 remains" reads as being about the post-renumbering index space; I verified the stronger property that surviving indices are exactly 0..8.
  - Only `model.safetensors` is required in `out/T1`, so I did not copy the config/tokenizer files from `inputs/base`.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: ~5 minutes.
