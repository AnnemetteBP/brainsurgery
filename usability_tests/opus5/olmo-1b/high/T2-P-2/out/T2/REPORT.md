# T2 — Participant self-report (condition P)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The input is sharded, so the state dict has to be reassembled from
    `model.safetensors.index.json` before anything can be sliced.
  - `lm_head.weight` sits alone in the second shard, which looked like a tied
    embedding; loading the shards separately keeps them as distinct tensors, and
    `.clone().contiguous()` on the pass-through tensors avoids any
    shared-storage rejection by `save_file`.
  - The q/k/v vs o_proj axis flip (row blocks vs column blocks) is the only real
    trap; I checked it by comparing the kept slices against the original on both
    sides of the seam rather than trusting the shapes alone.
  - Output is a single unsharded 5.0 GB file, so no index file is written.
- **Anything in the task text or documentation that was unclear:** nothing. The
  task gave exact keep-ranges and target shapes, which removed the usual
  ambiguity about head layout and ordering.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent:** ~5 minutes.
