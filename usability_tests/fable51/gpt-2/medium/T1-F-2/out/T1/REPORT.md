# T1 participant self-report (condition F)

- Final artifact path: `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Avoided the renumbering collision hazard entirely by building a fresh dict from an explicit old->new index map instead of renaming in place.
  - `attn.bias` is a causal-mask buffer, not a parameter; a naive "bias" filter would misclassify it, so the block regex targets the whole `h.<i>.` prefix.
  - Sanity load into a 9-layer `GPT2LMHeadModel` reports `h.*.attn.bias` as unexpected keys because transformers 5.x no longer registers that buffer; the task requires keeping them, so this is expected.
- Anything in the task text or documentation that was unclear: nothing; the required checks and mapping were explicit.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `load_file` / `save_file` for checkpoint I/O; the whole task is a key-level rewrite, so a plain script on top of it is the shortest and most auditable route.
  - `torch` 2.14.0: tensor equality for the fidelity checks.
  - `transformers` 5.12.1: post-hoc sanity check only (loading the output into a 9-layer config); not part of the solution script.
  - Considered mergekit passthrough slicing and torch-state-bridge, but both add config or rule authoring for a one-line mapping and would still need custom checks.
- Approximate time spent, if you can tell: about 3 minutes.
