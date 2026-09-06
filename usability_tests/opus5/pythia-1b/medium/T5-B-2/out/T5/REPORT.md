# T5 self-report (condition B, Pythia-1B)

- **Final artifact path:** `out/T5/plan.yaml` (output checkpoint in `out/T5/`:
  4 shards + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 3

- **Which executions failed, and why (one line each):**
  1. `crash` — `TransformError: matmul source_b missing: lora::base_model\.model\.gpt_neox.layers.0...` : I wrote `from_b` as a regex with escaped dots, but on a ternary transform `from_b` is a *rewrite template* of the `from_a` captures, so the backslashes were taken literally.
  2. `crash` — `TransformError: count.of matched zero tensors: base::.*lora_.*` : I expressed "no `lora_` tensor remains" as `count: {is: 0}`, but reference resolution raises on zero matches before the count is compared; had to use `not: { exists: ... }`.
  3. passed.

- **Pitfalls or surprises you hit (one line each):**
  - Ternary transforms (`matmul`, `add`, ...) take `from_a` as the matching pattern and `from_b`/`to` as `\1`-style rewrites — regex escaping belongs only in `from_a`.
  - A "must not exist" assertion cannot be written as `count: 0`; `not: { exists: }` is the only form that survives resolution.
  - `output` alias inference: since `delete` and in-place transforms count as writes, all intermediates had to be parked on the `base` alias (`base::<module>.mergedelta`) and the adapter alias left untouched, otherwise the run would have reported `cannot infer output model uniquely`. Conveniently this also means the adapter tensors can never leak into the output.
  - Doing the accumulation in float32 needed no extra tensors: `cast_` the 16 base weights to float32 in place, `add_` the delta, `cast_` back to float16 — the name and shape are preserved throughout.
  - Shard budget: `shard: 512MB` is binary (512 MiB) as documented, so the two 206 MB embedding tensors are *not* over budget and get packed together with neighbours rather than isolated; the task text's parenthetical suggested otherwise, but I followed the tool's documented packing rule (state-dict order, fill to budget, oversized tensor alone).

- **Anything in the task text or documentation that was unclear:**
  - The claim that `gpt_neox.embed_in.weight` and `embed_out.weight` (206 MB each) are "a single tensor larger than [512 MiB]" is wrong; they are well under budget. It left ambiguity over whether the grader expects them isolated. I went with the documented behaviour.
  - The README documents capture-based destination synthesis for binary transforms but not that `from_b` of a ternary transform is a rewrite rather than an independent pattern; the `add` example (`from_a: '.*.weight', from_b: '.*.delta'`) reads as if both are patterns.
  - `adapter_config.json` lists `target_modules: ["query_key_value"]` while TASK.md says `["attention.query_key_value"]`; immaterial here since only one module family is adapted.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading `help.txt` for the reference/rewrite semantics.
