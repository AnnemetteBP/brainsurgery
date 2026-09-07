# Participant self-report: T4 (GPT-2 124M), condition B

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 160 tensors; executed-plan summary in `out/T4/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - With three inputs the output alias is inferred from where the transforms write, so all scratch work (task vectors) had to live inside alias `base` (`base::tv1.<name>`, `base::tv2.<name>`) and be deleted before saving, rather than editing `ft1`/`ft2` in place.
  - `add`/`subtract` need a pre-existing destination, so the merge is expressed as `copy` (ft -> scratch), `subtract_` (scratch -= base), `scale_` (0.4), `add_` (base += scratch); both task vectors are formed before either is added to base, which avoids the ordering hazard.
  - Name-set equality is not a single assert; it is built from `count` (160 each, 112 non-MLP, 48 MLP per alias) plus `equal` with a negative-lookahead regex mapped onto `base::\g<0>`, which also enforces that every non-MLP name exists in base.
- Anything in the task text or documentation that was unclear:
  - The `add_`/`subtract_` help text does not say that `to` may be a capture rewrite of `from`; that is only in the interfaces reference ("Mapping note") and the README's `equal` description.
  - The README refers to a `docs/` tree that is not in the doc pack; the help pack was sufficient anyway.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 3 minutes (reading docs and writing the plan; the run itself took ~1 s of compute plus loading/saving).
