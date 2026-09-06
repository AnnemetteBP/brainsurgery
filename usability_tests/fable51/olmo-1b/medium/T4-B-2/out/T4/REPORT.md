# Participant self-report: T4 (Condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T4/plan.yaml` (output checkpoint: `out/T4/model.safetensors`, 114 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The in-place transforms (`add_`, `subtract_`) only document slice examples, not regex capture rewrites in `to`; the interfaces reference says mapping transforms support the same capture model, and it worked (`from: 'base::tv1\.(.+)', to: 'base::\1'`).
  - `subtract`/`add` require an existing destination, so the task vectors were built as scratch tensors inside the `base` alias (`copy` ft MLP -> `base::tv1.<name>`, then `subtract_` base, `scale_` 0.4) and deleted afterwards; keeping every write on `base` also keeps output-alias inference unambiguous.
  - Both task vectors were computed before either `add_` so that each is taken against the unmodified base.
  - A negative lookahead regex `(?!model\.layers\.\d+\.mlp\.).+` with `right: 'base::\g<0>'` handles the 66 non-MLP identity checks in one `assert: equal`.
- Anything in the task text or documentation that was unclear:
  - Whether `add_`/`subtract_` accept capture-group rewrites in `to` (only the ternary `add`/`subtract` and `copy`/`move` say so explicitly).
  - "Exactly 48 tensors were merged" has no direct counter in batch mode (`writes` needs an instrumented backend); expressed as `count` asserts on the 48 scratch task-vector tensors and the 48 MLP targets.
- Tools used (condition F): n/a (condition B, only `brainsurgery`).
- Approximate time spent, if you can tell: about 5 minutes reading the doc pack and writing the plan; the run itself took ~17 s.
