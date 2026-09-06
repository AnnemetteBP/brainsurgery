# T4 self-report (condition B: BrainSurgery plan)

- Final artifact path: `out/T4/model.safetensors` (plan: `out/T4/plan.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed
  all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - The ordering hazard is real: I avoided it by materialising `0.4*ft1` and
    `0.4*ft2` as scratch tensors *before* touching the base, then rewriting the
    merge as `0.2*base + 0.4*ft1 + 0.4*ft2` with `scale_` + two `add_`.
  - With three inputs the output alias must be unambiguous, so every scratch
    destination had to be written on the `base::` alias (`to: base::tv1__\1`),
    not on `ft1::`/`ft2::`.
  - Scratch names had to be prefixed (`tv1__...`) so that the later
    `scale_: target: 'base::h\.\d+\.mlp\..+'` (full-match regex) could not pick
    them up; `delete` removes them again before the output is written.
  - `assert: equal` with `left: 'base::(?!h\.\d+\.mlp\.).+'`, `right: 'ft1::\g<0>'`
    does the whole "everything outside the MLP is identical" check in one line,
    including the missing-name case, which was nicer than I expected.
  - Name-set equality is not directly assertable, so I pinned it with counts
    (160 total / 48 MLP / 112 non-MLP per alias) plus the 112 pairwise `equal`s.
- Anything in the task text or documentation that was unclear:
  - The docs do not state explicitly whether `add_`'s `to` supports capture-group
    rewriting the way `copy`'s `to` does; it does, but I had to infer it from the
    `equal` documentation ("like `to` in copy/move").
  - The task lists the MLP tensor names but not whether the checkpoint uses a
    `transformer.` prefix; I checked the safetensors header directly to confirm
    the bare `h.<i>....` names.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~15 minutes, mostly reading
  `docpack/help.txt`.
