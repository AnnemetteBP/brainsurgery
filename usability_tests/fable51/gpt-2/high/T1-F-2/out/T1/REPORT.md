# T1 self-report (GPT-2 124M, condition F)

- Final artifact path: `out/T1/solution.py` (produces `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `attn.bias` is a causal-mask buffer, not a parameter; a name-level regex that treats `.bias` as a parameter would still be correct here, but it is worth knowing when loading into HF (transformers 5.x reports the 9 `h.<i>.attn.bias` keys as unexpected because it no longer registers the buffer; they are kept in the output as the task requires).
  - Renumbering into a fresh dict (old index -> new index map, then explicit collision check) sidesteps the in-place move-order hazard entirely.
  - The "Required checks" list says "no tensor of blocks 9, 10, 11 remains"; I read that as output-name indices >= 9 must not exist and also check that indices are exactly 0..8 and that each block has 13 tensors.
- Anything in the task text or documentation that was unclear: nothing significant; see the note above on the block 9..11 check wording, which refers to output indices rather than the removed input blocks 2, 5, 8.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: load/save of the checkpoint (`load_file`/`save_file`), metadata `format=pt`.
  - `torch` 2.14.0: `torch.equal` for bit-exact value checks and `.contiguous()` before saving.
  - `transformers` 5.12.1: only for a post-hoc sanity check that the output loads into a `n_layer=9` GPT-2 config (not part of `solution.py`).
  - Plain Python `re` for the `^h\.(\d+)\.(.+)$` block pattern. I chose a script over mergekit/torch-state-bridge because the task is a 30-line dict rewrite, and a script lets me enforce the required checks, bit-exact value comparison and atomic write (temp file renamed only after reload verification) directly.
- Approximate time spent, if you can tell: about 3 minutes.
