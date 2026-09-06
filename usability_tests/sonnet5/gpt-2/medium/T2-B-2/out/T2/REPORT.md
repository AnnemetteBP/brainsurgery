# Participant self-report

- Final artifact path: `out/T2/plan.yaml` (output written to `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1 (plus one exploratory experiment run against a scratch plan in `out/tmp/`, deleted afterward, to confirm how `concat` resolves regex sources before writing the real plan)
- Which executions failed, and why (one line each):
  - The exploratory `concat` experiment failed on purpose: `concat`'s `from` list requires each entry to resolve to exactly one tensor, so a regex like `h\.(\d+)\.attn\.c_attn\.bias::[:320]` matches all 12 layers at once and errors with "concat source must match exactly one tensor, got 12" — this ruled out a single regex-driven concat across layers and confirmed the plan needs one explicit `concat` per layer per tensor.
  - The real plan (`out/T2/plan.yaml`) succeeded on its only run.
- Pitfalls or surprises you hit (one line each):
  - `concat` does not broadcast over regex captures the way `copy`/`move`/`assign` do (per the README's "pattern-based destination synthesis" note), so per-layer head slicing has to be written out explicitly for all 12 layers rather than as one regex rule.
  - `concat`'s destination must not already exist and `move`'s destination must not already exist either, so each tensor had to be rebuilt under a `.pruned` temp name, the original deleted, then moved back into place, rather than assigned in place.
  - Column/row boundaries for the kept segments came straight from the TASK.md required-result list (0-319/384-767 for q, 768-1087/1152-1535 for k, 1536-1855/1920-2303 for v, and 0-319/384-767 for the c_proj row blocks); slicing syntax is half-open `[a:b]` so `319` inclusive is written as `320` as the upper bound.
- Anything in the task text or documentation that was unclear: No — the task's explicit column/row lists made the slicing unambiguous, and `help: concat` / `help: move` in the doc pack were sufficient once the per-layer explicit-tensor requirement was discovered.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes including the exploratory doc/help review and one scratch experiment.
