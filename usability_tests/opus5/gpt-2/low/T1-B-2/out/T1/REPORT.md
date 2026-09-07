# T1 self-report (condition B)

- Final artifact path: `out/T1/plan.yaml` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - 1: `crash`/`no_match` — `assert: count: {of: 'h\.(2|5|8)\..*', is: 0}` raised "count.of matched zero tensors"; count cannot express "zero matches".
  - 2: `crash` — my replacement `assert: { not: { exists: { of: ... } } }` failed to parse; `exists` takes a bare reference string, not an `of` mapping.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: solved by issuing the 7 `move`s in ascending new-index order (3→2, 4→3, 6→4, 7→5, 9→6, 10→7, 11→8), so every destination is always free.
  - Every reference resolver errors on zero matches, so "nothing remains" must be written as `not: exists`, not `count: is: 0`.
  - `exists` payload shape differs from the other assert operators (bare string vs `of:` mapping).
- Anything in the task text or documentation that was unclear: the README's assert list does not show that a zero-match reference is an error for `count`, nor the bare-string payload of `exists`.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes.

## What the plan does

Asserts the input has 12 blocks / 160 tensors, deletes `h\.(2|5|8)\..*`, checks
those are gone, renumbers the survivors with 7 regex `move`s in ascending order,
then asserts: no `h.9/10/11.*` remain, exactly 9 `h.<i>.attn.c_attn.weight`,
exactly 121 tensors total, 117 block tensors (9 x 13) and the 4 non-block tensors
present. Output written to `out/T1/model.safetensors`.
