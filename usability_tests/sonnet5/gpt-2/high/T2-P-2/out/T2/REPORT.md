# Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to double-check that `c_attn`'s q/k/v segments are each 768 columns wide
    (2304 = 3 × 768) before computing per-segment head-block offsets, since the
    per-head slice inside each segment is head-index × 64, not a single global
    offset across the whole 2304-wide tensor.
  - Cross-checked the computed keep-index ranges against the exact ranges
    given in the task text (assertions in the script) before trusting the
    slicing logic, since a silent off-by-one in the block boundary would
    otherwise produce a checkpoint that loads but runs with garbage attention.
- Anything in the task text or documentation that was unclear: none; the
  per-tensor shapes and exact keep-ranges given in "Required result" were
  sufficient to implement directly.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes to write and verify
  the script.
