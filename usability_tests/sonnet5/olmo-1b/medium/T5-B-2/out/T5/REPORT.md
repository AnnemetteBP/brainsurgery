## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed.
- Pitfalls or surprises you hit (one line each):
  - `assert: { count: { ..., is: 0 } }` raises instead of succeeding, because the
    underlying resolver treats a zero-match reference as an error before `count`
    even compares the number; switched those checks to `assert: { not: { exists: ... } } }`,
    which does work for a "should not exist" check since `not` catches the resolver's error.
  - `matmul`'s destination must not already exist, so the `B @ A` product for
    each adapter pair has to land in a scratch tensor name (`...delta.weight`)
    rather than directly overwriting the base weight; fold it in afterwards with
    `add_` and then `delete` the scratch tensor so it doesn't leak into the
    sharded output.
  - Ternary transforms (`matmul`, `add`, ...) drive iteration from `from_a`'s
    matches and rewrite `from_b`/`to` from its capture groups, so `from_a`,
    `from_b` and `to` can each live under a different alias as long as the
    regex captures used in the rewritten refs (`\1`, `\2`) come from `from_a`.
- Anything in the task text or documentation that was unclear: none; the
  README's worked MoE-upcycling example (regex captures reused across aliases
  in `copy`) was enough to figure out the same pattern for `matmul`/`add_`.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: about 20 minutes, mostly reading
  `docpack/help.txt` for `matmul`/`scale_`/`add_`/assert payloads and one
  fix-and-rerun cycle for the zero-match `count` assertion.
