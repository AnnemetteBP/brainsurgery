# Participant self-report

- Final artifact path: `out/T4/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `subtract`/`add_`/`assign` all require the destination tensor to already exist, so
    intermediate task-vector tensors had to be created first with `copy` from the base
    (to get the right shape/dtype) before being overwritten.
  - Writing intermediates to a new alias (`work`) required an explicit
    `prefixes: { mode: add, alias: work }` transform first; an unregistered alias in a
    tensor ref fails plan compilation with "unknown model alias".
  - Using several aliases would make the implicit `output:` alias inference ambiguous
    ("cannot infer output model uniquely"), so the plan uses an explicit `save` transform
    with `alias: base` instead of an `output:` block.
- Anything in the task text or documentation that was unclear: none; the README's own
  example for the negative-lookahead "everything outside `h.<i>.mlp.*`" pattern mapped
  directly onto this task's precondition check.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: a few minutes of plan authoring plus one run.
