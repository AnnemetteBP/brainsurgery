# T3 — Participant self-report

- **Final artifact path:** `out/T3/solution.py` (driver: `out/T3/run.sh`)
- **Number of times you executed the script or plan:** 2 (the second was a
  no-op re-run after deleting an unused `import shutil`; the first execution
  already produced the correct output)
- **Which executions failed, and why:** none failed.
- **Pitfalls or surprises you hit:**
  - `transformers` `save_pretrained(dtype=...)` cannot express this task: it
    applies one dtype to the whole model, so mixed precision is out of reach;
    that ruled out the route `F-allowed.md` suggests for T3.
  - The obvious pattern `.*weight` would also hit `wte.weight`, `wpe.weight`
    and the 48 layer-norm weights, so I anchored the regex to the four
    projection names and escaped every dot.
  - `h.<i>.attn.bias` is a causal-mask buffer, not a bias parameter — a
    pattern on `attn.*bias` would delete `attn.c_attn.bias` too.
  - `wte.weight` is 147 MiB, well over the 64 MiB shard budget, so the packer
    has to allow a single oversized tensor to sit alone rather than erroring.
  - safetensors rejects non-contiguous tensors, so I call `.contiguous()`
    before saving.
- **Anything in the task text or documentation that was unclear:**
  - The shard *naming* scheme and the tensor iteration order used by the
    hidden reference are not stated. I used the HuggingFace convention
    (`model-0000i-of-0000n.safetensors`, greedy packing in the input file's
    key order) since that is what serving stacks expect.
  - Whether the index needs `metadata.total_size` is not stated; I included it
    because HF's loader and `transformers` both write it.
- **Tools used (condition F):**
  - `torch` 2.14.0 — dtype cast (`.to(torch.bfloat16)`, round-to-nearest-even
    as required) and bit-exact comparison in the verification pass.
  - `safetensors` 0.5.3 — `safe_open` to stream the input, `save_file` to
    write each shard.
  - Deliberately *not* used: `transformers` (single-dtype export only, see
    above), `mergekit` (its dtype conversion is also whole-checkpoint, and it
    is built around merge recipes, not selective casting),
    `torch-state-bridge` (this task renames nothing), `peft` (no adapters).
    A ~150-line script expresses the per-tensor rules directly and lets the
    required checks run before anything is written.
- **Approximate time spent:** roughly 5 minutes.

## How the required checks are enforced

`check()` runs on the in-memory state dict *before* any file is written and
raises `AssertionError` on: not exactly 48 bfloat16 tensors,
`h.0.attn.c_attn.weight` not bfloat16, `wte.weight` not float32, and a total
other than 148 tensors. It additionally rejects any non-float32 tensor outside
the projection set, any dropped non-buffer tensor, and any invented name.
`build()` fails if the number of deleted mask buffers is not 12.

After writing, `verify()` re-reads the shards from disk, re-runs `check()` on
what was actually persisted, confirms the index `weight_map` covers exactly
the output tensors, confirms every multi-tensor shard is within the 64 MiB
budget, and confirms every value is bit-exact against the input (cast where
the tensor is one of the 48). A failure at any point aborts with a non-zero
exit status.
