# Participant self-report: T5 (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/plan.yaml` (output checkpoint in `out/T5/`: 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - With two inputs, the output alias is inferred from where the transforms write, so the `matmul` intermediates had to be created on the `model` alias (not `lora`) and deleted afterwards to keep the output alias unambiguous.
  - The PEFT prefix `base_model.model.` had to be stripped via a regex capture group when mapping adapter names to base names.
  - `count`/`exists` assertions use full-match regex, so dots must be escaped and `.*lora_.*` is needed to catch any leftover adapter or intermediate name.
- Anything in the task text or documentation that was unclear:
  - The `matmul` help does not state how `from_b`/`to` are rewritten from `from_a` captures; I relied on the interfaces reference's "ternary mapping" note and it worked as expected.
  - The docpack `help.txt` does not include per-expression help (e.g. `count` keys `of`/`is`); the README covered it.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 5 minutes including reading the doc pack; the plan itself ran in ~18 s.
