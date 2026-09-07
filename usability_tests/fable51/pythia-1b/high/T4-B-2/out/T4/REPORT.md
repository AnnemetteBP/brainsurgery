# T4 (Pythia-1B), condition B: participant self-report

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, executed-plan summary `out/T4/summary.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all asserts and wrote 244 tensors.
- Pitfalls or surprises you hit (one line each):
  - With three inputs the output alias is inferred from write destinations, so every scratch tensor (float32 copies, task vectors) had to be created on the `base` alias (`to: 'base::t1.\1'` when casting from `ft1::...`) to avoid `cannot infer output model uniquely`.
  - `subtract`/`add` need a pre-existing destination, so the merge was done with `cast` (float32 working copies) + in-place `subtract_`/`scale_`/`add_`, then `cast_` back to float16 and `assign` into the original 64 names (keeps names and state-dict order), then `delete` of the scratch prefixes.
  - Ordering hazard handled by computing both task vectors against the untouched float32 base copy before any `add_` into it.
  - "Same tensor names" was expressed as: count 244 in all three, 180 non-MLP + 64 MLP in base, 64 MLP in ft1/ft2, and `equal` with `\g<0>` rewrite for every non-MLP tensor against ft1 and ft2 (which also fails if a name is missing on the right).
  - The 16 `attention.bias` mask buffers are uint8 (not float16); they are covered by the non-MLP `equal` checks and copied unchanged.
- Anything in the task text or documentation that was unclear:
  - The doc pack does not say what dtype a binary op produces when operands differ (e.g. f16 minus f16 into an f32 destination), so I avoided mixed-dtype arithmetic entirely by casting all operands to float32 first.
  - `assert.writes`/`reads` are described as "for instrumented backends" without saying which providers are instrumented, so the "exactly 64 merged" check uses `count` on the scratch/merged patterns instead.
- Tools used (condition F): n/a (condition B, brainsurgery only)
- Approximate time spent, if you can tell: about 3 minutes reading docs and writing the plan; the run itself took about 11 s wall clock.
