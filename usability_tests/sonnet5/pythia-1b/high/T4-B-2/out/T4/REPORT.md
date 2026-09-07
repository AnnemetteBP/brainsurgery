## Participant self-report

- Final artifact path: `out/T4/plan.yaml` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `crash` — `subtract`/`add` (ternary form) require an already-existing
    destination (README says "into existing destination(s)"; I initially misread it
    and tried to create new tensors with them), so it failed with
    `subtract destination missing: base::_scratch_d1.gpt_neox.layers.0.mlp.dense_4h_to_h.bias`.
    Fixed by cloning the destination first with `copy` and then using the in-place
    `subtract_`/`add_` variants.
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract` (non-`_` forms) need a pre-existing destination just like
    `assign`, unlike `copy`/`cast`/`scale` which require the destination to *not*
    exist; only `matmul` among the ternary ops creates new destinations.
  - With multiple inputs, every transform destination (across the whole plan, not
    just the final one) must resolve to a single alias or `brainsurgery` refuses to
    infer the output model, so all scratch/workspace tensors were kept as
    `base::_scratch_*` names rather than on their own alias, and deleted before
    saving.
  - The README's own example for excluding a subtree with a negative lookahead
    (`left: 'a::(?!h\.\d+\.mlp\.).+', right: 'b::\g<0>'`) maps almost verbatim onto
    the "verify everything outside the MLP tensors is identical" precondition, which
    made that check straightforward once found.
- Anything in the task text or documentation that was unclear:
  - Nothing in the task; the one point in the docs that cost an attempt was the
    "existing destination" requirement for `add`/`subtract`, which is stated once in
    the README table but easy to skim past next to `copy`/`cast`/`scale`, which are
    the opposite.
- Tools used (condition F): name, version, and why: n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes of interaction
