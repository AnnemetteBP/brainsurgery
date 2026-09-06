# T1 self-report (Condition B: BrainSurgery plan)

- **Final artifact path:** `out/T1/model.safetensors` (plan: `out/T1/plan.yaml`)

- **Number of times you executed the script or plan:** 2 (the second was a
  re-run to capture the exit code; both succeeded and produced the same output).
  One additional throwaway plan (`out/negcheck.yaml`, since deleted) was run to
  confirm that a failing `assert` exits non-zero and writes no output.

- **Which executions failed, and why (one line each):**
  - None. Both runs of `out/T1/plan.yaml` succeeded on the first try.
  - (Deliberate negative control: `out/negcheck.yaml` asserted `count 121` on the
    160-tensor input and correctly exited 1 with no file written.)

- **Pitfalls or surprises you hit (one line each):**
  - The collision hazard the task warns about is avoided by doing the renumbering
    moves in ascending source order (3->2, 4->3, 6->4, 7->5, 9->6, 10->7, 11->8);
    each destination slot is already vacated when it is written, and `move`
    refuses to overwrite an existing name, so a wrong order would fail loudly
    rather than silently clobber a block.
  - Regex block patterns need anchoring care: `h\.1\..*` would not have been safe
    against `h.10.*`/`h.11.*`, so every pattern includes the trailing `\.` before
    the rest of the name.
  - `move` accepts a regex with a capture group in `from` and a rewrite template
    (`h.2.\1`) in `to`, which turns the whole 13-tensor block rename into one
    transform per block; this is not spelled out under `help: move`, I inferred it
    from the `assert.equal` docs that describe `right` as "a rewrite ... exactly
    like `to` in copy/move".

- **Anything in the task text or documentation that was unclear:**
  - The README documents the `to` rewrite semantics only indirectly (under
    `assert.equal`); a one-line example of a capture-group rename in the `copy`/
    `move` help would have saved a lookup.
  - Nothing unclear in the task text itself; the old->new index mapping was given
    explicitly, which removed all ambiguity.

- **Tools used (condition F): n/a** (condition B: plan only, no Python written).

- **Approximate time spent:** ~10 minutes, most of it reading `docpack/README.md`
  and the `help.txt` entries for `move`, `delete` and `assert.count`.

## Verification performed

Beyond the in-plan asserts (no `h.9/10/11` tensors, exactly 9
`h.<i>.attn.c_attn.weight` matches, 117 block tensors, 121 tensors total, the 4
non-block tensors present), the output was checked against the input: key set
equals the expected renumbered key set, and every surviving tensor is
bit-identical in dtype, shape and values to its source.
