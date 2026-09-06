# Participant self-report: T1 (OLMo-1B-0724-hf, condition B)

- Final artifact path: `out/T1/plan.yaml` (output: `out/T1/model.safetensors`, 86 tensors)
- Number of times you executed the script or plan: 2 (first run succeeded; second run was a re-run only to confirm the exit code, same plan, same result)
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `move` requires the destination to be free, so the renumbering must be done in ascending old-index order (3->2, 4->3, ..., 15->11); done in one shot with the structured `*rest` variadic pattern per block.
  - No single-transform "renumber by expression" exists; ten explicit `move` transforms were needed.
- Anything in the task text or documentation that was unclear:
  - Required check says "no tensor of blocks 12..15 remains" while the removed blocks are 2, 6, 10, 14; I implemented it as written (post-renumbering indices 12..15 must be absent), plus a count of exactly 12 q_proj tensors and 86 total.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~3 minutes
