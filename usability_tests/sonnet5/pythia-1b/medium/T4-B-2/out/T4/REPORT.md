## Participant self-report

- Final artifact path: `out/T4/plan.yaml` (output written to `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `add`/`subtract` require the destination tensor to already exist, so each
    task-vector accumulator (`diff1`, `diff2`, `merged`) had to be pre-created
    with a `copy` before being used as a `to`.
  - For `add`/`subtract`/`cast`, `from_b`/`to` are not independent regexes —
    they are rewrite templates applied via `re.sub` over the `from_a` match
    (like `to` in `copy`), so they must use plain backreferences (`\1`, `\2`,
    `\3`) rather than repeating the regex syntax (e.g. `\.` or `(\d+)`), which
    otherwise raises "bad escape" errors.
  - Scratch tensors need an explicit `prefixes: { mode: add, alias: work }`
    before they can be targeted; an alias can't be referenced solely as a
    transform destination.
  - Doing all scratch arithmetic under a separate `work::` alias (rather than
    `base::`) kept `base` as the only alias written to, so the output-model
    alias could still be inferred unambiguously — solved by writing the final
    merged values back into `base` with `assign` and using an explicit `save`
    step instead of a top-level `output:` block.
- Anything in the task text or documentation that was unclear: The `help`
  text for `add_`/`subtract_` doesn't say whether they support regex-batched,
  paired source/destination tensors the way `add`/`subtract` explicitly do;
  I avoided the ambiguity by using the non-underscore forms with pre-created
  destinations.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~20 minutes, most of it verifying
  the `add`/`subtract` rewrite-template semantics against a small synthetic
  checkpoint before running on the real inputs.
