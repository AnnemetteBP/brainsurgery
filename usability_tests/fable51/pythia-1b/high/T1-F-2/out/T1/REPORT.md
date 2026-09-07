# T1 self-report (condition F, Pythia-1B)

- Final artifact path: `out/T1/solution.py` (produces `out/T1/model.safetensors`, 184 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None during the run. The collision hazard was avoided by building a fresh output dict from the source dict (never renaming in place) and failing on any duplicate destination name.
  - Loading the result into a 12-layer `GPTNeoXForCausalLM` (transformers 5.12.1) reports the 3 per-block buffers (`attention.bias`, `attention.masked_bias`, `attention.rotary_emb.inv_freq`) as unexpected keys, because current transformers no longer registers them; the task requires keeping them, so they are kept. All parameters load with no missing keys.
- Anything in the task text or documentation that was unclear:
  - The required check says "no tensor of blocks 12, 13, 14, 15 remains" while the removed blocks are 2, 6, 10, 14; I read the check as "no index >= 12 after renumbering" and enforced that, plus the stricter check that the block set is exactly 0..11.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 (`load_file` / `save_file`): direct, lossless single-file I/O; keeps float16/uint8 dtypes and buffers byte-exact.
  - `torch` 2.14.0: `torch.equal` for the bit-exact source-vs-output verification.
  - `transformers` 5.12.1: only for a post-hoc load into a 12-layer config, not part of the solution.
  - Did not use mergekit (passthrough slicing goes through HF model loading and would drop the non-parameter buffers and re-save with its own dtype handling) or torch-state-bridge (a 30-line regex remap did not need a framework). A plain script gave full control over the checks and atomic writing.
- Approximate time spent, if you can tell: about 3 minutes; the script runs in ~5 s.

## Checks enforced by `solution.py` (exit 1, nothing written, if any fails)

1. Input has 244 tensors and blocks exactly 0..15.
2. No destination-name collision while renumbering.
3. No block index >= 12 remains; block set is exactly 0..11.
4. Exactly 12 `gpt_neox.layers.<i>.attention.query_key_value.weight` tensors.
5. Exactly 184 output tensors.
6. Every kept tensor matches its source in shape, dtype and value (`torch.equal`).
7. Output is written to a temp path and renamed only after all checks pass; the written file is re-read and its key set compared.
