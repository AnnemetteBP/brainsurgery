# T1 — Depth pruning with layer renumbering (Pythia-1B), condition F

- **Final artifact path:** `out/T1/solution.py` (entry point `out/T1/run.sh`,
  run from the sandbox root). Output: `out/T1/model.safetensors`.
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the first execution produced the
  output and passed all checks.
- **Pitfalls or surprises you hit:**
  - `torch_state_bridge` applies its rules *sequentially to each already-rewritten
    key*, so a naive one-stage rule list (`layers.3. -> layers.2.`,
    `layers.4. -> layers.3.`, ...) can cascade a key through several rules; here it
    happens to be safe only because the rules are in ascending order and every
    destination index is lower than all later source indices. I did not rely on
    that: I renumber in two stages through a placeholder token
    (`gpt_neox.__block__<new>.`) that no source pattern can match, which makes the
    result independent of rule order, and asserted afterwards that no placeholder
    survived.
  - Its collision detector reports nothing for the naive rules precisely because the
    cascade rewrites keys instead of colliding — collision detection alone is not
    enough to catch this class of bug, so I also check the full expected key set,
    contiguous indices 0..11, and 15 tensors per block.
  - The three non-parameter buffers per block (`attention.bias` uint8,
    `attention.masked_bias`, `attention.rotary_emb.inv_freq`) must be carried over
    for the 184-tensor count to work out. That ruled out the "obvious" tool route
    (mergekit `passthrough` layer slicing / a `transformers` load + `save_pretrained`
    round trip): those rebuild the checkpoint through the model class, which does not
    reliably re-emit non-persistent buffers, and would not have been bit-exact.
- **Anything in the task text or documentation that was unclear:** nothing. The
  explicit old→new index table and the 184-tensor count made the target unambiguous.
  (The `query_key_value` head-interleaved row layout is described but is not needed
  for this task, since whole blocks move unchanged.)
- **Tools used (condition F):**
  - `torch-state-bridge` 0.1.0 — the actual renaming, via `state_bridge` /
    `state_bridge_preview` with `detect_collision=True`. Chosen because rule-based
    key rewriting with collision detection is exactly the shape of this task.
  - `safetensors` 0.5.3 — lazy `safe_open` read (only surviving tensors are
    materialised) and `save_file` write.
  - `torch` 2.14.0 — `torch.equal` for the bit-exact post-write verification.
  - Not used: `mergekit` 0.1.4 and `transformers` 5.12.1, for the buffer/bit-exactness
    reason above; `peft` is irrelevant here.
- **How the required checks are enforced:** all of them run in memory *before* any
  write, and each calls `die()` → non-zero exit with no output file. Checked: input
  has 244 tensors; exactly 60 tensors dropped; no block index ≥ 12 remains; exactly
  12 `attention.query_key_value.weight` tensors; indices are exactly 0..11; 15 tensors
  per block; the 4 non-block keys are unchanged; total is exactly 184; and the key set
  equals an expected set derived independently of the rule engine. After writing to
  `model.safetensors.tmp` the file is re-read and every tensor compared for dtype,
  shape and bit-exact value against its source tensor; only then is it `os.replace`d
  into `model.safetensors`. A failure at that stage deletes the temporary file.
- **Approximate time spent:** ~6 minutes, dominated by reading the
  `torch-state-bridge` source to confirm its rule-application semantics.
