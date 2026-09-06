# T5 self-report (condition B, BrainSurgery plan)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `output` writes exactly one alias, inferred from which alias the transforms write to, so all intermediates had to live in the `base` alias (`__delta.*`) and not in `lora`, otherwise the plan would fail with "cannot infer output model uniquely".
  - `matmul` resolves `from_b` and `to` as `re.sub` rewrites of each `from_a` match, so one regex with a capture group merges all 16 layers in a single transform.
  - `matmul` rejects mixed dtypes, and the base is float16 while the adapter is float32; casting the target weights in place to float32 (`cast_`), `add_`, then `cast_` back to float16 gives the required float32 accumulation.
  - `assert: count` cannot express "zero matches" — resolution raises on zero matches before the count is compared — so the "no `lora_` in the output" check is `assert: { not: { exists: ... } }`.
  - Escaped dots matter: `\d+` patterns with unescaped `.` would also match e.g. `dense_4h_to_h`-style neighbours; every pattern here is fully anchored and dot-escaped.
- Anything in the task text or documentation that was unclear:
  - The task says `embed_in.weight` / `embed_out.weight` (206 MB each) are "stored alone in its own shard", but they are well under the 512 MiB budget, so the tool's oversize rule never triggers for them; I read the sentence as describing the general rule, not this checkpoint.
  - The README doc-pack links point at absolute paths on the author's machine (`/Users/petersk/...`), so the linked `docs/` pages are not reachable from the sandbox.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~10 minutes, most of it reading `help.txt` and the transform source to confirm rewrite and dtype semantics before the first run.

## Verification done

Independent check after the run (not part of the plan): all 244 names present,
4 shards each ≤ 512 MiB of tensor data, no `lora_`/`__delta` names in the index,
228 unchanged tensors bit-identical to the base, and the 16 merged weights match
`(W.float() + 2.0 * B @ A).half()` exactly (relative Frobenius error 0.0).
