# T1 self-report (Pythia-1B, condition F)

- Final artifact path: `out/T1/solution.py` -> `out/T1/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The classic renumbering collision was avoided by building a fresh output dict
    keyed by new names instead of renaming in place; a `dest_exists` check on the
    new key still guards it explicitly.
  - Block regex is anchored on `^gpt_neox\.layers\.(\d+)\.` with escaped dots so
    it cannot reach the 4 non-block tensors (`embed_in`, `embed_out`, final LN).
  - Non-parameter buffers (`attention.bias` uint8 mask, `masked_bias`,
    `rotary_emb.inv_freq`) are carried through like any other block tensor; no
    dtype coercion anywhere, so float16 and uint8 survive bit-exact.
  - `save_file` rejects shared/non-contiguous storage, so tensors are passed
    through `.contiguous()` on write.
- Anything in the task text or documentation that was unclear: nothing; the
  explicit old->new index table removed all ambiguity. The QKV row-layout detail
  is irrelevant for T1 since no tensor is sliced.
- Tools used (condition F):
  - `safetensors` 0.5.3 — load/save of the checkpoint; the task is purely a key
    rewrite, so direct state-dict I/O is the smallest correct route.
  - `torch` 2.14.0 — tensor handles, contiguity, shape/dtype equality checks.
  - Considered and rejected: `mergekit` passthrough layer slicing would work but
    needs a full HF model dir round-trip and would rewrite/re-dtype metadata it
    owns, risking non-bit-exact output; `torch-state-bridge` regex rewriting is a
    reasonable fit but adds a dependency for ~10 lines of mapping logic and still
    leaves persistence and the required checks to me.
- Approximate time spent, if you can tell: ~5 minutes.

## Checks enforced by the script (all before any write)

Each check calls `fail()`, which prints to stderr and exits non-zero without
writing output:

1. No tensor of blocks 12..15 remains.
2. Exactly 12 tensors match `gpt_neox.layers.<i>.attention.query_key_value.weight`.
3. Surviving block indices are exactly the contiguous set 0..11.
4. The output has exactly 184 tensors.
5. No renumbering collision (a new key is never written twice).
6. Every output tensor has the shape and dtype of its source tensor.
