# Participant self-report: T2 (OLMo-1B-0724-hf head pruning, condition B)

- Final artifact path: `out/T2/plan.yaml` (output checkpoint: `out/T2/model.safetensors`, 114 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed_assertion / no_match. My extra sanity check `assert: { count: { of: 'pruned\..*', is: 0 } }` raised "count.of matched zero tensors" because `count` treats zero matches as an error rather than counting 0. All required edits and required checks had passed before it; no output was written. Replaced with `assert: { not: { exists: 'pruned\..*' } }`.
  - Execution 2: success.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so the per-layer slicing cannot be expressed with one regex/capture rewrite; I generated 64 explicit `concat` entries (16 layers x q/k/v/o) with a shell loop into the YAML.
  - Pruned tensors have to go to temporary names first (destinations must not exist), then `delete` the originals and `move` the temporaries back with a capture-group rewrite.
  - `count` cannot assert "zero matches"; use `not: exists` instead.
- Anything in the task text or documentation that was unclear:
  - The README does not say that `count` with `is: 0` is an error on zero matches.
  - No documented way to slice-and-concat across a pattern of tensors in one transform.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
