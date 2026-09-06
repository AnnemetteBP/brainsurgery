# Participant self-report: T5 (GPT-2 124M), condition B

- Final artifact path: `out/T5/plan.yaml` (executed summary in `out/T5/summary.yaml`; output shards and `model.safetensors.index.json` in `out/T5/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed every assertion and wrote 5 shards.
- Pitfalls or surprises you hit (one line each):
  - With two inputs the output alias is inferred from where the transforms write, so all intermediates (`tmp.h.<i>.delta`, `tmp.h.<i>.delta_t`) had to live on the `base` alias and be deleted before saving; nothing may be written to `lora`.
  - `matmul` only writes to new destinations and `add_` only to existing ones, so the merge is matmul -> `scale_` -> `permute` (transpose for `fan_in_fan_out=true`) -> `add_` -> `delete`, rather than one step.
  - Regex references need escaped dots on the match side (`h\.(\d+)\.`) but plain dots on the rewrite side (`h.\1.`), following the doc-pack example.
  - The sharder placed `wte.weight` (154 MB, over the 100 MiB budget) alone in the last shard rather than the first; the plan cannot influence shard order.
- Anything in the task text or documentation that was unclear:
  - The README does not say explicitly whether a ternary transform's `from_b` may use `\1` from `from_a`; the interfaces reference's "capture-based rewrite model across from_a, from_b, and to" line implied it, and it worked.
  - No documented assert counts "adapter pairs merged" directly; I approximated it with `count` on the adapter A and B patterns, `count` on the temporaries after `matmul`, and a `writes >= 1` check on the 12 merged base weights.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 3 minutes (doc reading, one plan run, one verification pass).
