# T5 self-report (condition F, OLMo-1B-0724-hf)

- Final artifact path: `out/T5/solution.py` (output checkpoint: `out/T5/model-0000{1..10}-of-00010.safetensors` + `out/T5/model.safetensors.index.json`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 succeeded but packed `model.embed_tokens.weight` (412 MB, under the 512 MiB budget) together with a 64 MB tensor; I changed the rule so tensors over half the budget are stored alone, as the task describes for the two big tensors, and re-ran.
- Pitfalls or surprises you hit (one line each):
  - The task says the 412 MB embedding/lm_head tensors are "larger than" the 512 MiB budget; they are not, so pure greedy packing does not isolate them. I added a half-budget "store alone" threshold to match the stated expected layout.
  - `adapter_config.json` lists `target_modules` as `q_proj`/`v_proj` (no `self_attn.` prefix) while TASK.md quotes the prefixed form; I matched by suffix.
  - PEFT adapter names carry the `base_model.model.` prefix which must be stripped to reach the base names.
- Anything in the task text or documentation that was unclear:
  - Whether the grader checks the exact shard layout or only the per-shard budget and index consistency. Both of my runs satisfy the budget; the final run also stores the two large tensors alone.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: float32 `B @ A`, scaling and addition.
  - `safetensors` 0.5.3: lazy shard reading, `save_file` for the output shards.
  - Plain Python for the index and greedy sharding. I did not use `peft`/`transformers` `merge_and_unload` + `save_pretrained`: instantiating the model is unnecessary for a file-level merge, and a script gives direct control over the 512 MiB budget, bit-exactness of untouched tensors, and the required checks.
- Approximate time spent, if you can tell: about 5 minutes.

Independent verification after the final run: all 82 unchanged tensors are bit-exact against the base; the worst relative Frobenius error of the 32 merged weights against a float64 reference is 5.2e-8.
