# T4 self-report (condition B, Pythia-1B)

- Final artifact path: `out/T4/plan.yaml` (output `out/T4/model.safetensors`);
  `out/T4/check.yaml` is an extra, independent verification plan that writes nothing.
- Number of times you executed the script or plan: 1 execution of `out/T4/plan.yaml`
  (plus 1 execution of the separate verification plan `out/T4/check.yaml`).
- Which executions failed, and why (one line each): none; the plan passed on the first run.
- Pitfalls or surprises you hit (one line each):
  - Output-alias inference: every write must land on one alias, so the float32 scratch
    tensors from `ft1`/`ft2` were created on the `base` alias (`cast: from 'ft1::(...)' to 'base::d1.\1'`)
    rather than cast in place on the fine-tune aliases.
  - Ordering hazard: both task vectors are taken against the untouched base by accumulating
    into scratch `acc.*` tensors and writing the base slots only once at the end (`assign`),
    so the second task vector never sees an already-merged base.
  - Escaping dots in the scratch prefixes matters (`base::d1\.(gpt_neox\..+)`), otherwise the
    pattern would also reach into unrelated names.
  - `subtract`/`add_` resolve `to`/`from_b` as a rewrite of the driving pattern's captures,
    so all 64 tensors are handled by one transform each.
- Anything in the task text or documentation that was unclear:
  - The task says "verify that the three checkpoints have the same tensor names"; the assert
    language has no direct set-equality operator, so name identity is established by
    counting (244 total, 180 non-MLP, 64 MLP with the layer index bounded to 0..15 and the
    four exact suffixes enumerated) plus `equal` on the 180 non-MLP tensors, which also
    requires each name to exist in the other checkpoint.
  - "computed in float32, then cast back to float16" leaves open whether the intermediate
    task vectors must be float32 as well; everything after the initial cast is float32 here.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes, most of it reading `docpack/help.txt`;
  the plan itself runs in about 12 seconds.
