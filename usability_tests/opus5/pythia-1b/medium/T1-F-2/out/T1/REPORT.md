# T1 (Pythia-1B depth pruning) — participant self-report

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution passed all checks.
- **Pitfalls or surprises you hit:**
  - `torch-state-bridge` applies its rules *sequentially* to each key, not
    simultaneously, so a naive per-layer rule list can rewrite a key twice
    (old 4 -> 3 -> 2). I ordered the rules ascending by source index, which
    makes every destination strictly below every later source, and left
    `detect_collision=True` on as a backstop.
  - Literal rule patterns needed care against prefix aliasing: `layers.1.`
    must not match `layers.13.`. Anchoring on the trailing dot
    (`gpt_neox.layers.1.`) makes this safe, and the per-tensor bit-exactness
    check below would have caught it anyway.
  - The three non-parameter buffers (`attention.bias` uint8,
    `attention.masked_bias`, `attention.rotary_emb.inv_freq`) are ordinary
    keys here, so no special handling was needed — but they do have to be
    counted in the 15-tensors-per-block bookkeeping.
  - `safetensors` rejects shared/non-contiguous storage, so the tensors are
    written `.contiguous()`.
- **Anything in the task text or documentation that was unclear:** whether
  `out/T1` should also contain a 12-layer `config.json`. The "Required result"
  says a single file `model.safetensors` and grading compares the key set, so
  I wrote only the checkpoint.
- **Tools used (condition F):**
  - `torch-state-bridge` 0.1.0 — rule-based key rewriting with built-in
    collision detection, which is exactly the hazard of this task (block
    renumbering overwriting a survivor). Chosen over mergekit's passthrough
    layer slicing because mergekit would re-export a full HF model and does
    not give bit-exact control over the buffer tensors.
  - `safetensors` 0.5.3 — load/save of the checkpoint.
  - `torch` 2.14.0 — `torch.equal` for the bit-exactness verification.
- **Approximate time spent:** ~5 minutes.

## How the required checks are enforced

`solution.py` exits non-zero (via `die()`) and writes no output if any of:
input tensor count != 244; pruned count != 184; a key collision is detected
during renumbering; output count != 184; block indices != 0..11; any tensor of
blocks 12–15 survives; `query_key_value.weight` count != 12; block tensor
count != 180; the non-block key set differs or any of those four tensors
changed; or any renumbered tensor is not bit-, shape- and dtype-identical to
the source tensor of the block it came from. The file is written to a
temporary path and `os.replace`d into position only after every check passes,
then re-read and re-verified.
