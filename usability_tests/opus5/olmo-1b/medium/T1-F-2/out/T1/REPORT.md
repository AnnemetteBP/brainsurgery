# T1 (condition F) — participant self-report

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - `torch_state_bridge.RuleEngine.apply` runs *every* rule in sequence on each
    key, so a direct rule list (`layers.3.→layers.2.`, `layers.4.→layers.3.`, …)
    cascades old block 4 down to index 2; its `detect_collision` would not even
    fire, because the cascade produces distinct-but-wrong names. I renumbered in
    two passes through a temporary namespace (`model.__pruned__.<new>.`) so each
    key matches at most one rule per pass.
  - The library's capture placeholders only match digits, so the renumbering map
    (which is not affine: 0,1,3,4,5,7,… → 0..11) had to be written as 12 literal
    rules rather than one arithmetic rule.
  - Literal rules like `model.layers.3.` are substring (unanchored) matches, but
    they are safe here because the trailing dot and the `model.layers.` prefix
    pin the whole index segment — `model.layers.13.` does not contain them.
  - The input is sharded (2 shards + index), the output is a single file; I read
    via the index `weight_map` and checked the loaded tensor count against it.
- **Anything in the task text or documentation that was unclear:** nothing
  material. The required checks are stated in terms of blocks 12..15 remaining;
  I additionally asserted that the surviving indices are exactly 0..11 and that
  every renamed tensor is bit-identical to its source, which is stricter.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — `safe_open` to stream the two input shards and
    `save_file` to write the single output file; the natural I/O layer here.
  - `torch-state-bridge` 0.1.0 — rule-based key rewriting with collision
    detection, which is the part of the task with the real hazard (the bulk
    rename). Used for both renumbering passes.
  - `torch` 2.14.0 — only as the tensor framework behind safetensors
    (`.contiguous()` before saving).
  - I did *not* use mergekit's passthrough layer slicing: its slice syntax
    builds a model from index ranges, and expressing "drop 2, 6, 10, 14" needs
    five slices plus a full `save_pretrained` round trip, which risks dtype or
    key-name drift against a bit-exact grader. A script over safetensors keeps
    the tensors untouched by construction (the checks assert identity, not just
    equality, of the tensor objects passed through).
- **Approximate time spent:** ~5 minutes, most of it reading the
  `torch-state-bridge` source to confirm its rule-application order.

## Checks enforced by the run

`solution.py` exits non-zero and writes no output if any of these fails:
input tensor count vs index; exactly 28 tensors dropped for blocks 2/6/10/14;
collision detection on both rename passes; output has exactly 86 tensors;
surviving block indices are exactly 0..11 (so nothing from 12..15 remains);
exactly 12 tensors match `model.layers.<i>.self_attn.q_proj.weight`; both
non-block tensors present and passed through unchanged; every survivor's
renamed key exists and holds the original tensor; and a readback of the written
file matches the intended key set (the file is deleted if it does not).
