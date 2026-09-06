## Participant self-report

- Final artifact path: `out/T1/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none — block boundary is a prefix match on `h.<i>.`, so removing/renumbering by regex-captured index is safe against accidental partial matches (e.g. `h.1.` vs `h.10.`) as long as the match is anchored at the start and followed by a literal dot.
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: a few minutes
