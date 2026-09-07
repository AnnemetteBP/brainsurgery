# T4 self-report (condition B, BrainSurgery plan)

- **Final artifact path:** `out/T4/plan.yaml` -> `out/T4/model.safetensors`
  (160 tensors, float32, single file).

- **Number of times you executed the script or plan:** 3 executions of
  `out/T4/plan.yaml` (1 failed, 2 succeeded — the third was a re-run of the
  already-correct plan to confirm the output file and the log).
  Separately, 5 executions of throwaway plans under `tmp/` that never write
  output: 1 `dump` to read the tensor names, 3 small probe plans to pin down
  transform semantics (2 of those failed), and 1 verification plan that
  re-derives the merge and compares it to `out/T4/model.safetensors`.

- **Which executions failed, and why (one line each):**
  - `out/T4/plan.yaml` #1 — `failed_assertion` / `no_match`: my own
    `assert: { count: { of: 'base::tv[12]\..*', is: 0 } }` after the scratch
    deletes; the reference resolver raises `count.of matched zero tensors`
    before `count` can compare, so `is: 0` is not expressible that way.
    Replaced with `assert: { not: { exists: 'base::tv[12]\..*' } }`.
    Everything before it — the whole preflight and the merge — had passed.
  - `tmp/probe1.yaml` — `crash`: `add_ invalid source regex 'a.\1'`. I had
    guessed that `add_` is driven by its `to` pattern; the error proved the
    opposite and told me what I needed.
  - `tmp/probe2.yaml` #1 — `crash`: `missing model alias in reference: 'a.x'`
    (with several inputs there is no default alias; every ref needs `alias::`).

- **Pitfalls or surprises you hit (one line each):**
  - The ordering hazard the task warns about is real and easy to get wrong: I
    materialise *both* task vectors (`tv1`, `tv2`) fully before folding either
    one into `base`, so both are taken against the unmodified base.
  - Direction of the pattern rewrite is not the same across transforms and is
    not spelled out in the docs: `add_`/`subtract_` are driven by `from` (with
    `to` as the rewrite, like `copy`), while `subtract`/`add` are driven by
    `from_a`. I probed this on 2-element scratch tensors rather than guess.
  - `count: { is: 0 }` cannot express "nothing matches" — the resolver errors
    on zero matches first. `not: { exists: ... }` is the way, and it works
    because `not` catches exactly that `TransformError`.
  - Everything must be written to one alias or the run fails with
    `cannot infer output model uniquely`; staging the task vectors as
    `base::tv1.<name>` (not on `ft1`) keeps the output unambiguous.
  - Scratch names must not collide with the real pattern. `tv1.h.0.mlp.…`
    does not full-match `h\.\d+\.mlp\.…`, so the merge patterns stay clean —
    but this only holds because matching is `re.fullmatch`, not `search`.
  - Checking "the three checkpoints have the same tensor *names*" is awkward,
    since `equal` compares values. I get it from counts plus a one-to-one
    `equal` mapping: 112 non-MLP names matched into each fine-tune, 48 MLP
    names counted in each, and 160 total in each — so the name sets coincide.
  - The negative lookahead `(?!h\.\d+\.mlp\.)(?s:.+)` with `right: '…::\g<0>'`
    is the compact way to say "every tensor except the MLP ones"; the README
    documents this idiom and it worked first try.

- **Anything in the task text or documentation that was unclear:**
  - The docs do not state which reference drives pattern matching for the
    in-place (`add_`, `subtract_`) and three-reference (`add`, `subtract`)
    transforms; the per-command `help` shows only literal-name examples. This
    cost me two probe runs.
  - The task says "computed in float32" but not whether the accumulation order
    is constrained; the grading tolerance (1e-5 relative Frobenius) answers it.
  - Not documented: with more than one input there is no default alias, so
    even scratch tensors created by `ones`/`zeroes` need an explicit `alias::`.

- **Tools used (condition F):** n/a — condition B, plan only, no Python written.

- **Approximate time spent, if you can tell:** ~5 minutes.

## What the plan does

1. **Preflight (must pass before anything is touched).** 160 tensors in each of
   `base`/`ft1`/`ft2`, split 48 MLP / 112 non-MLP in each; all 112 non-MLP
   tensors bit-identical between `base` and each fine-tune (`equal` also
   enforces matching shape and dtype); float32 on the MLP tensors; the four
   MLP shapes on layer 0.
2. **`tv1 = 0.4*(ft1 - base)`** and **`tv2 = 0.4*(ft2 - base)`**, both staged as
   `base::tv1.*` / `base::tv2.*` and both computed against the untouched base.
3. **`base += tv1; base += tv2`** over the 48 MLP tensors only.
4. **Cleanup and postconditions.** 48 tensors in each task vector and 48 merge
   targets (the "exactly 48 merged" check), scratch tensors gone, 160 tensors
   in the output, still 48 MLP / 112 non-MLP, and the 112 non-MLP tensors still
   bit-identical to the inputs.

## Independent verification (not part of the deliverable)

`tmp/verify.yaml` re-loads `out/T4/model.safetensors` alongside the three
inputs and recomputes the merge by a different algebraic route,
`0.2*base + 0.4*ft1 + 0.4*ft2`, then asserts it equals the written result
within `eps: 1e-5`. It also re-checks the 160-tensor count, the bit-exact
non-MLP tensors, and that the merged tensors genuinely differ from the base.
All ten of its assertions passed.
