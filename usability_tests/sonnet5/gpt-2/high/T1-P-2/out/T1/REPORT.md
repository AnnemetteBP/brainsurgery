# Participant self-report

- Final artifact path: `out/T1/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single
  execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None encountered; the tensor naming (`h.<i>.<rest>`) was unambiguous, so
    a single regex on the leading `h.<idx>.` prefix cleanly separated block
    vs. non-block tensors without touching `attn.bias` or other `.` bearing
    suffixes.
  - Built the old-index -> new-index remap up front from the sorted list of
    surviving indices, then wrote into a fresh dict rather than mutating in
    place, which avoids the renumbering-collision hazard described in the
    task (e.g. shifting block 3 into slot 2 before block 2's tensors are
    removed) by construction.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
