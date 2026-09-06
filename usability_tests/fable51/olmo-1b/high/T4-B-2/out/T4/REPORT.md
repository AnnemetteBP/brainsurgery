# T4 participant self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`,
  executed-plan summary `out/T4/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `subtract`/`add` (ternary) require the destination to already exist, so the task
    vectors were built with `copy` (ft -> scratch name on `base`), then in-place
    `subtract_` and `scale_`, instead of a single `subtract` into a new tensor.
  - Output alias inference with several inputs: every write had to land on `base`
    (scratch task vectors were created as `base::tv1.<name>` / `base::tv2.<name>` and
    deleted before saving) so that `base` is the unique output alias.
  - Ordering hazard handled by computing both task vectors against the untouched base
    before either `add_`; a `count` of 48 per scratch set is asserted before merging.
  - Name-set equality expressed without a dedicated operator: per-alias `count` of 114
    total and 48 MLP tensors, plus `equal` of the 66 non-MLP base tensors against
    `ft1::\g<0>` and `ft2::\g<0>` (which fails if a right-hand name is missing).
- Anything in the task text or documentation that was unclear:
  - `docpack/help.txt` does not state that assert `of` references (`shape`, `dtype`,
    `count`) accept patterns; the worked example plan showed `shape` with a regex, and
    it worked for `dtype` too.
  - Whether `\g<0>` is accepted in `to` of `copy`/`subtract_` was only documented for
    `assert.equal` ("exactly like `to` in copy/move"); it works.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes of reading and writing;
  the plan runs in about 35 s (in-memory provider, CPU).
