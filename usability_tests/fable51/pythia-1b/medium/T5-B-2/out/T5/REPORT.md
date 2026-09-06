# Participant self-report: T5 (Pythia-1B, condition B)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`, 4 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `no_match` — `matmul source_b missing: lora::base_model\.model\...\.lora_A\.weight`. I wrote `from_b` as an escaped regex with a `\1` backreference; the capture-rewrite for `from_b`/`to` is a literal template, so the backslash-escaped dots were kept as literal characters and matched nothing.
- Pitfalls or surprises you hit (one line each):
  - In ternary transforms only `from_a` is a regex; `from_b` and `to` are literal rewrite templates (captures `\1` work, but dots must not be escaped there).
  - `add_` between a float32 delta and a float16 base: rather than rely on implicit promotion I did `cast_` to float32, `add_`, `cast_` back to float16, matching the task's "compute in float32, cast back" wording.
  - Adapter tensors live on a separate `lora` alias, so they never reach the output; only the `base` alias is written (all writes target `base::`), so the output alias inference is unambiguous.
  - Intermediate deltas were created on `base` with a name containing `lora_delta` and then deleted, so the `not exists '.*lora_.*'` check also guards against leaving them behind.
- Anything in the task text or documentation that was unclear:
  - The docs do not say explicitly that `from_b`/`to` rewrite templates are literal (not regex); the `add` help example (`from_b: '.*.delta'`) suggested otherwise.
  - The task says embed_in/embed_out (206 MB) are "larger than" the 512 MiB budget and stored alone; they are not larger, and the tool packed them with other tensors. I followed the tool's packing rule (state-dict order, 512MB budget).
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes
