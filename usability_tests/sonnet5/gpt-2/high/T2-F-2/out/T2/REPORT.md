# Participant self-report

- Final artifact path: `out/T2/solution.py` (invoked via `out/T2/run.sh`), output at `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - GPT-2's Conv1D layout means c_attn heads are column blocks and c_proj heads are row blocks, not the other way around as for nn.Linear — matched against the task spec's explicit block ranges before writing code, rather than trusting a mental model of "Linear" pruning.
  - The fused c_attn tensor holds three independent 768-wide q/k/v segments; the per-head slice has to be repeated with the same head-5 offset (320:384) inside each of the three segments, not just once globally.
- Anything in the task text or documentation that was unclear: none — the exact column/row ranges to keep were given directly in the spec, which made the transform a straightforward index-select rather than something requiring interpretation.
- Tools used (condition F): `safetensors` 0.5.3 (streaming read of the input file, write of the output) and `torch` 2.14.0 (index_select for the column/row slicing). Chose a plain script over `transformers.prune_heads` because the task spec pins the exact keep-ranges and tensor names already; a direct index-select against those ranges is less indirection and easier to verify bit-exact than round-tripping through a model class's own pruning implementation (which internally reindexes and isn't guaranteed to reproduce the exact block order requested here). `mergekit` and `peft` were considered but don't offer sub-tensor head-slicing at this granularity.
- Approximate time spent, if you can tell: a few minutes
