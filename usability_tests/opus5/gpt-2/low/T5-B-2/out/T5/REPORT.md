# T5 self-report

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`)
- Number of times you executed the script or plan: 3
- Which executions failed, and why (one line each):
  - 1: `crash` — `matmul source_b missing: lora::base_model\.model\.h\.0\...lora_A\.weight`; `from_b` is resolved as a *rewrite* of the `from_a` match (like `to`), so regex escapes must not be used there.
  - 2: `no_match` — `count.of matched zero tensors: base::.*lora_.*`; `assert count ... is: 0` cannot express "nothing matches", the reference resolver errors before the count is compared.
- Pitfalls or surprises you hit (one line each):
  - `from_b` of `matmul` uses rewrite syntax, not match syntax (backreference `\1` in a literal name).
  - "no tensor named `lora_`" had to be re-expressed as `count(of: 'base::(?!.*lora_).*', is: 160)` combined with the total count of 160.
  - Conv1D `[in, out]` vs Linear `[out, in]`: `permute: order [1, 0]` on `B @ A` before `add_`.
  - Intermediates must live in the `base` alias, otherwise the output alias cannot be inferred; they are removed with a single `delete` before writing.
  - Sharding is automatic: `shard: 100MB` is binary and counts tensor data only, so `wte.weight` (154 MB) lands alone in its own shard.
- Anything in the task text or documentation that was unclear: the docs do not state that `from_a`/`from_b` differ in reference semantics for binary transforms, and give no idiom for asserting the absence of a pattern.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~10 minutes
