# T2 self-report (condition B, BrainSurgery plan)

- Final artifact path: `out/T2/plan.yaml` -> `out/T2/model.safetensors` (114 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so the 16 layers
    could not be handled by one pattern; the plan enumerates 64 concat/delete/move triples.
  - Destination tensors must not exist, so each pruned tensor is built under a `pruned.` name,
    the original is deleted, and the pruned tensor is moved back to the original name.
  - o_proj holds heads on columns (dim 1) while q/k/v hold them on rows (dim 0); the
    `[out, in]` Linear layout makes that the only asymmetry.
  - Regex references need escaped dots (`model\.layers\.0\....`) to avoid overmatching.
- Anything in the task text or documentation that was unclear: nothing blocking; the README does
  not state that `concat` sources must be single-match, that only shows up in `help: concat`.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes.
