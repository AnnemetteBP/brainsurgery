# Participant self-report: T1 (GPT-2 124M, condition P)

- Final artifact path: `out/T1/solution.py` (output checkpoint `out/T1/model.safetensors`, 121 tensors, blocks 0..8)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Avoided the renumbering-collision hazard by building a fresh dict from an explicit old->new index map rather than renaming in place.
  - Anchored the block regex to `^h\.(\d+)\.` so `attn.bias` / `mlp.c_proj` names could not be mis-parsed; non-block names are checked against an explicit allowlist.
  - All checks (removed indices absent, exactly 9 contiguous `attn.c_attn.weight`, 121 tensors, per-block completeness and bit-equality) run before `save_file`, so a failure leaves no output.
- Anything in the task text or documentation that was unclear: the required check "no tensor of blocks 9, 10, 11 remains" reads oddly since the removed blocks are 2, 5, 8; I interpreted it as "no output name carries an index >= 9", which is the meaningful post-renumbering condition, and enforced it that way.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes.
