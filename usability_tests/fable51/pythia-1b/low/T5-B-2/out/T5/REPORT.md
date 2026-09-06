# Participant self-report: T5 (Pythia-1B, condition B)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 5 shards + `model.safetensors.index.json`, 244 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `no_match` — `matmul source_b missing: lora::base_model\.model\...\.lora_A\.weight`. I wrote `from_b` as an escaped regex with a `\1` backreference; `from_b` is resolved as a rewrite of each `from_a` match (like `to`), so the backslash-escaped dots were taken literally. Writing it as a plain name with `\1` fixed it.
- Pitfalls or surprises you hit (one line each):
  - In binary transforms (`matmul`, `add_`), only the first reference is a regex; the second and the destination are rewrite templates of the match, so they must not be regex-escaped.
  - Merging in float32 with in-place `add_` on a float16 target has undocumented dtype behaviour, so I built a float32 copy with `cast`, added, then `cast_` back to float16, deleted the original and `move`d the result over the original name (`move` requires the destination not to exist).
  - Two input aliases: all writes had to be on `base::` so the output alias is unambiguous; the `lora::` alias is never written and thus never exported.
- Anything in the task text or documentation that was unclear:
  - The `help` for `matmul`/`add`/`add_` says "references may be regex" but does not say that `from_b`/`to` are rewrite templates of `from_a` rather than independent regexes; only the `assert.equal` help explains this rewrite semantic.
  - dtype promotion rules for mixed-dtype `add_`/`matmul` are not documented.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
