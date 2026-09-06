# Participant self-report — T4 (condition B)

- Final artifact path: `out/T4/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution
  of `out/T4/plan.yaml` succeeded.
- Pitfalls or surprises you hit (one line each):
  - `copy`/`subtract`/`add_` destinations must live on an already-declared
    input alias — you cannot spawn a fresh scratch alias mid-plan (confirmed
    with a throwaway probe plan outside `out/`, which failed with
    `unknown model alias`). Worked around this by reusing `ft1`'s and `ft2`'s
    own MLP tensor slots as scratch space for the (scaled) task vectors,
    since only the `base` alias is ever saved.
  - In `subtract`/`add`/`multiply`, `from_a` is the regex that does the
    matching and captures groups; `from_b`/`to` must be *rewrite templates*
    of those captures (e.g. `model.layers.\1.mlp.gate_proj.weight`), not
    independent regexes — writing `from_b` as another `\d+`-style regex
    raised `bad escape \d` from `re.sub`. Verified the correct form with a
    2-tensor synthetic probe checkpoint before touching the real inputs.
  - Ordering hazard called out in the task text is real: since `subtract`
    needs an existing destination and I reuse `ft1`/`ft2` slots for scratch,
    it was important to compute *both* task vectors (against the untouched
    `base`) before either `add_` mutates `base` — doing the adds in between
    would have used a partially-merged base for the second task vector.
  - Used `save: { path, alias: base }` instead of the top-level `output:`
    block, since the plan also writes to the `ft1`/`ft2` aliases (in-place
    scratch use) and `output`'s alias-inference would otherwise fail with
    "cannot infer output model uniquely".
- Anything in the task text or documentation that was unclear: none; the
  worked `olmo_1b_0724_hf_dense_to_expert_moe.yaml` example in the doc pack
  was very close to what was needed here (regex batch `copy` across all 16
  layers with `\1` capture rewrites, cross-alias `equal` asserts).
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~20 minutes, most of it building
  and running two small synthetic-tensor probe plans outside `out/` to pin
  down `subtract`'s regex-rewrite semantics before writing the real plan.
