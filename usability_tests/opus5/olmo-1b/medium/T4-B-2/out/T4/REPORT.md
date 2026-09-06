# T4 self-report (condition B, BrainSurgery plan)

- **Final artifact path:** `out/T4/plan.yaml` -> `out/T4/model.safetensors`
  (114 tensors, float32, single file). Verification plan: `out/check_T4.yaml`
  (read-only, no `output:` block, writes nothing).

- **Number of times you executed the script or plan:** 2 executions of
  `out/T4/plan.yaml` (1 failure, then success). Plus 1 execution of the
  separate read-only verification plan `out/check_T4.yaml`.

- **Which executions failed, and why (one line each):**
  1. Execution 1 — `failed_assertion` / `no_match`: my final "no scratch left"
     check was written as `count: { of: 'base::tv\..+', is: 0 }`, but a
     tensor reference that matches zero tensors is rejected by the resolver
     before `count` compares, so it raised
     `count.of matched zero tensors: base::tv\..+`. Everything before it
     (the three-checkpoint verification and the whole merge) had already
     succeeded. Replaced with `not: { exists: 'base::tv\..+' }`.
  2. Execution 2 — success.

- **Pitfalls or surprises you hit (one line each):**
  - `count: { is: 0 }` cannot express "nothing matches"; the zero-match guard
    in the resolver fires first, so absence must be written as
    `not: { exists: ... }`.
  - Ordering hazard of the task: the second task vector must be taken against
    the *unmodified* base. I avoided it by accumulating both task vectors into
    one scratch set first (`acc = ft1 + ft2 - base - base`, then `acc *= 0.4`)
    and only touching `base` in a single final `add_`, so no in-place write to
    `base` happens before both vectors are formed.
  - All writes have to land on one alias or the run fails with
    `cannot infer output model uniquely` — so the scratch tensors are created
    on the `base` alias under a `tv.` prefix and deleted before the output is
    written.
  - Distinguishing "everything except MLP" needed a negative-lookahead regex
    (`(?!model\.layers\.\d+\.mlp\.).+`) with `right: '<alias>::\g<0>'`; the
    `equal` operator resolving `right` as a rewrite of each `left` match makes
    the 66-tensor cross-checkpoint comparison a single assert.
  - The `tv.` prefix on scratch names is deliberately not matched by the
    MLP pattern (full-match regex), so the later `subtract_`/`scale_` steps
    could not accidentally pick scratch tensors up.
  - Name-set agreement across the three checkpoints is enforced partly by
    counts (114 / 48 in each) and partly structurally: the `copy`, `add_` and
    `subtract_` rewrites require the identically-named tensor to exist in the
    other checkpoint, so a differing MLP name aborts the run.

- **Anything in the task text or documentation that was unclear:**
  - The docs do not say that a zero-match reference is an error in every
    context, which is what broke the `count: is: 0` form; worth a note next to
    `assert.count`.
  - "verify that the three checkpoints have the same tensor names" is not
    directly expressible as one assert (`equal` compares values too, and the
    MLP tensors legitimately differ); I had to combine count assertions with
    the structural name requirement of the rewrites.
  - Output sharding behaviour: I relied on the README's statement that a path
    with a `.safetensors` suffix is written as a single file rather than being
    sharded by the 5GB default, which is right at the boundary here (the file
    is 5.1 GB / 4.77 GiB). It did write one file.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `docpack/help.txt`; the two plan executions took ~8 s and ~15 s.

## Verification performed

`out/check_T4.yaml` re-derives the merge in a different order
(`0.2*base + 0.4*ft1 + 0.4*ft2`, algebraically identical to
`base + 0.4*(ft1-base) + 0.4*(ft2-base)`) and asserts, all passing:

- output has exactly 114 tensors, 48 of them MLP, all `float32`;
- the 66 non-MLP tensors are bit-identical to the base (no `eps`);
- the 48 merged tensors match the independent recomputation within
  `eps: 1e-5` absolute;
- `not: equal` — the merged MLP tensors did actually change vs. the base.
