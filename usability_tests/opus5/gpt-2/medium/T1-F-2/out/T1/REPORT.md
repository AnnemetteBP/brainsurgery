# T1 participant self-report

- **Final artifact path:** `out/T1/solution.py` (output: `out/T1/model.safetensors`, 121 tensors)

- **Number of times you executed the script or plan:** 2 (both succeeded).
  Execution 1 produced a correct output; execution 2 was a rerun after I
  tightened two things in the script (an order-preservation check, and
  removing a stale output at the start of a run instead of on failure).
  Two extra runs of *modified copies* under `out/T1/.selftest/` (since deleted,
  and pointed at a scratch output dir) were negative tests to confirm the
  required checks actually fire; they were not attempts at producing the output.

- **Which executions failed, and why:** none. The negative test with
  `DROP = (2, 5)` failed as designed (`failed_assertion`: "tensors of blocks
  >= 9 remain"), which is what I wanted to see.

- **Pitfalls or surprises you hit:**
  - `torch-state-bridge` applies its rules *sequentially per key*, so a naive
    rule list (`h.3.->h.2.`, `h.4.->h.3.`, ...) cascades: a key already
    renumbered by an earlier rule can be matched again by a later one. I
    routed every rename through a temporary `H.<new>.` namespace and mapped
    `H.` back to `h.` in a final rule, which makes each key hit exactly one
    renumbering rule.
  - Its `{n}` captures are digits-only (`\d+`), so the non-numeric tail of a
    key cannot be captured; the per-index rules use literal prefixes instead.
  - The 13th tensor per block is the causal-mask buffer `h.<i>.attn.bias`,
    which is easy to overlook if you think in terms of parameters rather than
    tensors; matching whole blocks by `^h\.(\d+)\.` picks it up automatically.
  - The three required checks (no blocks >= 9, exactly 9 blocks, 121 tensors)
    are all satisfied by a *reversed* renumbering as well — my negative test
    with the survivor list reversed passed them. That is exactly the silent
    failure the task warns about, so I added a check that the survivor order
    is ascending, plus a per-tensor comparison of every output tensor against
    the source tensor it is supposed to come from (shape, dtype, values).
  - Writing is atomic (temp file + `os.replace`) and a pre-existing output is
    removed at the start, so a failing run can never leave a stale or partial
    `model.safetensors`.

- **Anything in the task text or documentation that was unclear:** the task
  says "a single file `out/T1/model.safetensors`", so I wrote only that and no
  `config.json` with `n_layer: 9`, even though the objective mentions loading
  into a 9-layer configuration. Grading compares tensors only, so this seems
  right, but it was a judgement call.

- **Tools used (condition F):**
  - `safetensors` 0.5.3 — load/save of the checkpoint; the input and required
    output are both single safetensors files, so this is the direct route.
  - `torch-state-bridge` 0.1.0 — the block-index rewrite as declarative rules
    with built-in collision detection, which is the hazard this task is about.
  - `torch` 2.14.0 — tensor comparison in the verification checks.
  - Considered and rejected: `mergekit` passthrough layer slicing. It is the
    other suggested route, but it works at the level of HF model dirs and
    writes a re-serialized model (its own shard layout, config, extra
    metadata), which makes bit-exactness and "exactly one file with exactly
    121 tensors" harder to guarantee than to do the rename directly.
    `transformers` was unnecessary: nothing here needs a model graph.

- **Approximate time spent:** roughly 10 minutes, most of it reading the
  `torch-state-bridge` source to learn its rule semantics and writing the
  negative tests.
