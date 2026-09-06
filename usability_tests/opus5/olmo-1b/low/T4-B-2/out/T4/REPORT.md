# T4 self-report (condition B)

- Final artifact path: `out/T4/plan.yaml` -> `out/T4/model.safetensors` (114 tensors, float32, single file)
- Number of times you executed the script or plan: 1 (plus one separate verification plan in `/tmp`, not the task plan)
- Which executions failed, and why (one line each): none; the single run of the plan succeeded with all asserts passing.
- Pitfalls or surprises you hit (one line each):
  - Output-alias inference: every write must land on one alias, so the two task vectors had to be materialized as temporary tensors *inside* the `base` alias (`copy` from `ft1::`/`ft2::` to `base::tv1.\1`/`tv2.\1`) instead of scaling the fine-tune aliases in place.
  - Ordering hazard: both task vectors are computed (copy, `subtract_` against base, `scale_` by 0.4) before either `add_` touches the base, so the second vector is taken against the unmodified base.
  - Negative lookahead in the `left` reference plus `right: 'ft1::\g<0>'` is what makes the 66 non-MLP tensors checkable in one assert; the count assert next to it pins the 66/48 split so the lookahead cannot silently under-match.
  - Temporary names had to be deleted before output, and the final `count: 114` assert is what catches a leftover.
- Anything in the task text or documentation that was unclear:
  - The help text for `add_`/`subtract_` does not say which side (`from` or `to`) drives the pattern match and which is the rewrite; I assumed `from` matches and `to` is rewritten, as for `copy`. It worked, but the docs should state it.
  - `subtract`/`add` (non-in-place) require an already existing destination, which rules out the obvious "compute delta into a fresh name" form; only `copy`-then-`subtract_` works.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes, most of it reading the doc pack; the plan itself runs in ~35 s.
