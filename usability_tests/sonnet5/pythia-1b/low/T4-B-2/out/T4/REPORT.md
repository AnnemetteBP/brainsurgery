## Participant self-report

- Final artifact path: `out/T4/plan.yaml` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 2 (first attempt failed at
  plan-compile time; second attempt succeeded)
- Which executions failed, and why (one line each):
  - Attempt 1: `PlanLoaderError: unknown model alias: 'out'` — tried to write
    the merge into a brand-new `out::` alias, but a plan can only reference
    model aliases known from `inputs` or already inferred as an output alias
    from an earlier transform; a never-before-seen destination alias is
    rejected at compile time. Fixed by writing everything back onto the
    `base` alias (which already holds all 244 tensors) instead of inventing
    a separate output alias, so `base` is the single alias the plan writes
    to and can be inferred/used as the output.
- Pitfalls or surprises you hit (one line each):
  - `subtract`/`add_`/`scale_`/`assign` all require their destination to
    already exist, so each of the two task vectors (`ft1 - base`,
    `ft2 - base`) had to be pre-created via `copy` from a same-shaped scratch
    tensor before `subtract` could write into it.
  - Both task vectors have to be computed against the *original* base, so the
    float32 scratch copy of `base`'s MLP tensors was kept untouched until
    both deltas were computed and scaled, and only then accumulated into it
    with two `add_` calls (order doesn't matter for grading, but not
    re-deriving `base32` from an already-modified copy matters for
    correctness).
  - `cast`/`copy` destinations must not already exist, and `assign` requires
    matching dtype and shape, so the merged float32 result had to be cast to
    a new float16 scratch tensor first, then `assign`ed onto the real MLP
    tensor names (which already existed as the original float16 base
    values), then all scratch tensors (`_base32_*`, `_ft1_32_*`, `_ft2_32_*`,
    `_delta1_*`, `_delta2_*`, `_merged16_*`) were deleted so the final output
    has exactly 244 tensors, not 244 + scratch.
- Anything in the task text or documentation that was unclear: no; the
  `equal` assert's regex-rewrite semantics (documented via the
  `(?!pattern$).+` negative-lookahead example) mapped directly onto excluding
  the 64 MLP tensors from the shared-tensor check.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: one plan-compile failure plus one
  successful run; a few minutes of iteration overall.
