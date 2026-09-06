# Participant self-report: T5 (GPT-2, condition B)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`: 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 4
- Which executions failed, and why (one line each):
  - 1: `no_match` — `matmul` "source_b missing": regex-escaped dots (`\.`) in the `from_b`/`to` rewrite strings are taken literally, so the synthesized name did not exist.
  - 2: `failed_assertion` — my own `writes: is: 1` check on the merged weights fired; loading a tensor counts as a write, so the merged tensors had writes=2 (merge itself was correct).
  - 3: `no_match` — `assert count: { is: 0 }` errors with "matched zero tensors" instead of passing; replaced with `not: { exists: ... }`.
  - 4: success.
- Pitfalls or surprises you hit (one line each):
  - Rewrite side of a regex reference (`to`, `from_b`, `right`) is a literal template, not a regex: use plain dots there.
  - `count ... is: 0` cannot express "nothing matches"; `not: exists` is the way.
  - With several inputs the output alias is inferred from write targets, so intermediates must be created on the base alias (then deleted) rather than on the adapter alias.
  - No `transpose` transform; `permute` with `order: [1, 0]` does it.
  - Access counters include the initial load (writes start at 1), useful for asserting that non-merged tensors are untouched (`writes lt 2`).
- Anything in the task text or documentation that was unclear:
  - The docs do not say that rewrite strings are literal templates (dots unescaped) nor that `count is: 0` / zero-match references raise.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
