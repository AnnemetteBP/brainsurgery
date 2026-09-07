# T4 report

## Participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - None functionally; the main care point was ordering — computing each fine-tune's
    delta against the unmodified `base` tensors held in memory, not against a
    partially-merged output, since both deltas are added in the same pass.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — direct load/save of the three `model.safetensors` files
    (plain GPT-2 checkpoint files, not HF Hub repos with sharded index files, so
    `safe_open`/`save_file` is the direct, minimal-dependency path).
  - `torch` 2.14.0 — tensor arithmetic in float32, `torch.equal` for the
    bit-exact shared-tensor verification, `contiguous()` before saving.
  - Considered `mergekit`'s task-arithmetic YAML config, but it targets HF-Hub-style
    model directories/architectures and adds indirection (config resolution,
    architecture detection) with no accuracy or safety benefit for three flat
    safetensors files with a known, fixed tensor layout; a ~120-line script is
    easier to audit end-to-end against the three required checks.
- Approximate time spent, if you can tell: ~10 minutes.
