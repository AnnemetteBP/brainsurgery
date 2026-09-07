# T4 self-report (condition B)

- Final artifact path: `out/T4/plan.yaml` -> `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1 (succeeded on the first execution)
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - `cast_` takes `to:` for the dtype, while `cast` takes `dtype:`; I checked the
    help before running instead of guessing.
  - Output alias inference: with three inputs every write (scratch tensors
    included) has to be on one alias, so the float32 working copies of the ft1/ft2
    MLP tensors are cast *into* the `base` alias and deleted again at the end.
  - Ordering hazard: both task vectors are formed by subtracting the still-pure
    `acc.*` copy of the base before anything is added to it.
  - Name-set equality is not a single assert; I got it from three `count: 244`,
    three `count: 64` on the MLP pattern, and two `equal` checks over the
    complement (each `left` match must resolve to an existing `right` tensor).
- Anything in the task text or documentation that was unclear:
  - The README does not say whether `cast` may read from one alias and write to
    another; it works, but I only learned that by running.
  - Nothing states which dtype intermediate arithmetic uses, hence the explicit
    float32 scratch copies.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes, plan runtime 10 s.
