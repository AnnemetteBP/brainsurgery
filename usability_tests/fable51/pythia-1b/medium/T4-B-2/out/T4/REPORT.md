# T4 self-report (condition B, Pythia-1B)

- Final artifact path: `out/T4/plan.yaml` (output checkpoint: `out/T4/model.safetensors`, 244 tensors; executed-plan summary in `out/T4/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Ternary transforms (`subtract`, `add`) require the destination to already exist, so the merge is built from `cast` (which creates new float32 tensors) followed by in-place `subtract_` / `scale_` / `add_`.
  - With three inputs, the output alias is inferred from where transforms write, so every temporary (`work.*`, `d1.*`, `d2.*`) and the final `assign` had to be placed on the `base` alias.
  - Ordering: both task vectors are subtracted from the untouched float32 base copy before either scaled vector is added back, so ft2's delta is not taken against an already-merged base.
  - The "exactly 64 merged" check is expressed as count asserts on the 64 working tensors and on the 64 MLP names before/after the merge, plus a final `equal` check that the 180 non-MLP tensors are still bit-identical to the fine-tunes (and hence the base).
- Anything in the task text or documentation that was unclear:
  - The help text for `subtract_`/`add_` does not say explicitly that `to` is a regex rewrite of `from` with `\1` capture groups; I inferred it from `copy`/`assign`/`equal` and it worked.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes (reading the doc pack, one plan write, one run).
