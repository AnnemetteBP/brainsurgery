# T1 participant self-report (condition F, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T1/solution.py` (entry point `out/T1/run.sh`);
  output at `out/T1/model.safetensors`, 86 tensors, 4045416024 bytes.

- **Number of times you executed the script or plan:** 3 (2 attempts to first
  success, plus one idempotency re-run afterwards)

- **Which executions failed, and why (one line each):**
  1. `failed_assertion` — my own post-transform check "tensors of dropped blocks
     [2, 6, 10, 14] remain" fired: I had written a dropped-block name check
     against the *old* numbering but applied it to the *renumbered* output,
     where indices 2, 6 and 10 are valid names again for old blocks 3, 8 and 13.
     No output was written. Fixed by moving that check to load time, where the
     old numbering is still in force.
  2. Success.
  3. Success — a re-run of the identical script to confirm it is idempotent and
     reproduces the same output; it rewrote `model.safetensors` with the same
     result. Counted here because it did produce the output.

- **Pitfalls or surprises you hit (one line each):**
  - `torch_state_bridge.state_bridge` applies rules **sequentially, feeding each
    rule's output into the next**, so a naive one-pass renumber cascades: a key
    renamed 4->3 is picked up again by a later 3->2 rule. I confirmed this on
    dummy data — descending rule order raises a collision, ascending order only
    works by accident of ordering. I avoided depending on that by renaming
    through a scratch namespace (`model.__pruned__.<new>.`) that no rule source
    can match, making the rewrite order-independent.
  - "No tensor of the dropped blocks remains" is **not** checkable by name on the
    output: after renumbering, indices 2, 6 and 10 are legitimately occupied.
    The property that actually matters is old-key -> new-key *routing*, which I
    check by object identity (`out[new] is src[old]`) for all 86 tensors. This
    cost me execution 1.
  - Rule sources in torch-state-bridge are `re.escape`d, so the usual
    unescaped-dot overreach (`model.layers.3.` matching `model.layers.13.`)
    is not a hazard here — worth knowing rather than guessing.
  - The input is sharded across two files, so keys must be resolved through
    `model.safetensors.index.json`; loading only the 86 survivors avoids
    materialising the 4 dropped blocks at all.

- **Anything in the task text or documentation that was unclear:**
  - The required check "no tensor of blocks 12, 13, 14, 15 remains" reads as if
    it were about the *dropped* blocks, but blocks 12/13 are survivors (old 12
    and 13 become new 9 and 10). It only makes sense as a check on the *output*
    numbering, i.e. that nothing sits at an index >= 12. I implemented it that
    way. This ambiguity is close to what caused my one failed execution.
  - Required result 4 says the output is a single `model.safetensors`, while the
    objective says the result "must load into a 12-layer configuration". I wrote
    only the tensor file and did **not** emit a 12-layer `config.json`, since
    the required result is explicit about a single file and grading compares
    tensors only.

- **Tools used (condition F): name, version, and why:**
  - `torch-state-bridge` 0.1.0 — the key rewriting itself, in two rule passes
    with `detect_collision=True` on both. Chosen because the task's stated
    hazard is exactly a renumbering collision and this library detects that
    condition natively rather than leaving it to me to notice.
  - `safetensors` 0.5.3 — `safe_open` to read the two shards selectively
    (skipping the dropped blocks entirely) and `save_file` to write the result.
  - `torch` 2.14.0+cu130 — tensor identity, shape/dtype and `Tensor.equal`
    comparisons in the verification pass.
  - Considered and rejected: **mergekit** 0.1.4 passthrough layer slicing
    (`[0,2) [3,6) [7,10) [11,14) [15,16)`) would renumber automatically, but it
    emits a sharded directory under its own writer and default dtype policy,
    which fights the "single file, 86 tensors, bit-exact float32" requirement.
    **transformers** `save_pretrained` was rejected for the same reason — it
    reshards and would need a fabricated 12-layer config to load at all.

- **How the required checks are enforced:** all checks run *before* anything is
  written, and the payload goes to a `.tmp` file that is re-opened, re-verified
  against the source (key set, shapes, dtypes, `torch.equal` on all 86 tensors)
  and only then `os.replace`d into position — so a failing run exits non-zero
  with no output. I verified each required check fires by feeding `check_output`
  doctored dictionaries (stale block 12–15 present; 11 blocks; 87 tensors; a
  collision duplicating a block) — all four raised.

- **Independent verification:** a separate script re-read
  `out/T1/model.safetensors` and compared it against the original shards:
  86 tensors, exact key set, blocks 0..11 contiguous, 12 `q_proj` tensors, no
  index >= 12, all values bit-identical and float32, and the negative control
  that new layer 2 equals old layer 3 and *not* old layer 2.

- **Approximate time spent, if you can tell:** roughly 10 minutes wall clock,
  of which the three executions were ~17 seconds in total.
