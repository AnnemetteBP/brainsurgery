# Participant self-report: T5 (GPT-2 124M), condition B

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 5 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `matmul source_b missing: lora::base_model\.model\.h\.0\.attn\.c_attn\.lora_A\.weight` (no_match). In ternary transforms `from_b` is a rewrite template of the `from_a` match, not a regex, so the escaped dots I wrote were taken literally.
- Pitfalls or surprises you hit (one line each):
  - `from_b` in `matmul` (and `to`) is a rewrite of `from_a` with `\1` captures; it must use plain dots, while `from_a` uses an escaped regex.
  - With two inputs, every write (matmul/permute/scale destinations, `add_`, `delete`) has to target the `base::` alias so the output alias can be inferred; intermediates therefore live in `base::tmp_merge.*` and are deleted before the asserts and the write.
  - Transpose is expressed as `permute` with `order: [1, 0]`; there is no dedicated transpose transform.
- Anything in the task text or documentation that was unclear:
  - The README/help state that ternary transforms use "capture-based rewrite" across `from_a`/`from_b`/`to`, but not explicitly that `from_b` is a template (not a pattern); an example with `\1` in `from_b` would have avoided the failed attempt.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes
