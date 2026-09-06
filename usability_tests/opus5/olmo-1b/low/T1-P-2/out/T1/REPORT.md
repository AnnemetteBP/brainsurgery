# T1 — Participant self-report (condition P)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: avoided by building the whole renamed dict in a fresh
    mapping (with an explicit collision check) instead of renaming in place.
  - Input is sharded, so the state dict had to be merged from both shard files via
    `model.safetensors.index.json` before any renaming.
  - The layer regex is anchored and requires digits (`^model\.layers\.(\d+)\.`) so it cannot
    touch `model.embed_tokens.weight` / `lm_head.weight`.
- Anything in the task text or documentation that was unclear:
  - The task asks for a single output file but the input directory also holds config and
    tokenizer files; I read "Output: a single file `out/T1/model.safetensors`" literally and
    copied no config, and left `num_hidden_layers` untouched since no config output was asked for.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes.
