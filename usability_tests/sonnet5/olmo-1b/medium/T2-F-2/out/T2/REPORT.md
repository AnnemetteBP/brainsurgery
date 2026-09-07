# T2 participant self-report

- **Final artifact path:** `out/T2/solution.py` (writes `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single run succeeded.
- **Pitfalls or surprises you hit:**
  - `transformers.PreTrainedModel.prune_heads` (the route `F-allowed.md` suggests
    for T2) is only wired up for a handful of encoder models via `_prune_heads`
    hooks; OLMo does not implement it, so it wasn't usable here without extra
    plumbing. A direct safetensors slice-and-reassemble script was simpler and
    made the row/column block boundaries and piece order explicit and auditable.
  - Otherwise no surprises: q/k/v heads are row blocks, o_proj heads are column
    blocks, and `model.safetensors.index.json` cleanly maps every key to its
    shard.
- **Anything in the task text or documentation that was unclear:** No.
- **Tools used (condition F):**
  - `torch` 2.14.0 — tensor slicing (`index_select`) and dtype handling.
  - `safetensors` 0.5.3 — reading the two input shards via `safe_open` and
    writing the single merged output via `save_file`.
  - Chose a plain script over `mergekit`/`torch-state-bridge`/`transformers.prune_heads`
    because the transform here is a fixed, per-layer row/column slice on four
    named tensors — a direct script is the smallest, most auditable way to do
    that shape-critical operation and to assert the required shape/count checks
    before writing.
- **Approximate time spent:** A few minutes (single attempt, no retries).
