## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 5
- Which executions failed, and why (one line each):
  - #1: `PlanLoaderError: missing model alias for structured reference` — structured list-form tensor references (`["base", "delta", "$i", ...]`) have no way to carry an alias prefix when the plan has more than one input; had to switch to regex-form references.
  - #2: `TransformError: matmul invalid regex rewrite from ...: bad escape \d`  — `matmul`'s `from_b` is not an independent regex match, it is a rewrite template applied to the `from_a` match (like `to` in `copy`), so it can't contain its own `(\d+)`/`\.` regex syntax; rewrote `from_b` as a literal template using `\1`.
  - #3/#4: `PlanLoaderError`/`TransformError` around the "no adapter tensor left" checks — `assert: count` raises instead of returning 0 when a pattern matches zero tensors, and `assert: exists` expects the reference directly as the payload, not under an `of:` key; switched both checks to `assert: { not: { exists: <ref> } }`.
  - #5: succeeded.
- Pitfalls or surprises you hit (one line each):
  - Structured (list-form) tensor references silently require a single-input plan (`default_model`); with two inputs (`base`, `lora`) they fail to compile, so regex references with `alias::` prefixes are the only option for a two-checkpoint merge.
  - `matmul`/`add`/`subtract`/`multiply` are "ternary mapping" transforms: only `from_a` is matched against real tensor names; `from_b` and `to` are rewrite templates over `from_a`'s captures, so they must be literal names with `\1`-style backreferences, not their own regexes.
  - `assert: count` treats "zero matches" as a hard error rather than a valid count of 0, so "prove nothing matches" has to be written as `assert: { not: { exists: ... } }`.
  - To keep the output alias inferable, all intermediate tensors (`B@A`, scaled, cast) had to be materialized under the `base` alias (not `lora`), since every write-like transform (including `delete`) counts toward alias inference and touching two aliases makes the run fail with "cannot infer output model uniquely".
  - `output.shard: 512MB` maps to exactly 536,870,912 bytes per the binary-unit convention in the README, matching the task's byte budget with no unit conversion needed.
- Anything in the task text or documentation that was unclear:
  - The doc pack's help text for `matmul`/`add`/etc. doesn't spell out the capture-rewrite relationship between `from_a`/`from_b`/`to`; that's only mentioned in passing under "Mapping note" in `interfaces-reference.md` and had to be inferred from an error message.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~15 minutes including doc reading and iteration.
