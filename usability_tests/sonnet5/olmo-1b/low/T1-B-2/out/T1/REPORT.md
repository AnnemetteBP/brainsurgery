## Participant self-report

- Final artifact path: `out/T1/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The renumbering mapping always shifts old indices down or keeps them the
    same, so processing the `move` steps in ascending old-index order is
    sufficient to guarantee every destination is already vacated (by an
    earlier `delete` or an earlier `move` in the same list) before the next
    move writes to it; no reverse-order or temporary-name juggling was needed.
  - `move` requires the destination not to already exist, which doubles as a
    free collision check: if the ascending-order assumption were wrong, the
    plan would abort loudly instead of silently overwriting a block.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): n/a (condition B, plan-only).
- Approximate time spent, if you can tell: a few minutes, mostly reading the
  README's tensor-reference/regex-capture syntax and the doc pack's OLMo
  example plan before writing the plan.
