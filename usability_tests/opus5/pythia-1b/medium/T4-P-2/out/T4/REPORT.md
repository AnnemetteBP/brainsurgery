# T4 participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the first execution
  succeeded ("verified 180 shared tensors, merged 64 MLP tensors ... 244 tensors").
- **Pitfalls or surprises you hit (one line each):**
  - The ordering hazard is easy to get wrong: both task vectors must be taken
    against the original base, so I compute `b32` once per tensor and never
    write back into it before the second term.
  - `inputs/` also contains a `lora/` directory that is irrelevant to this task;
    I ignored it.
  - Used `safe_open` lazy access rather than `load_file` for all three
    checkpoints so only a few tensors are resident at a time during the
    verification pass.
  - `save_file` rejects non-contiguous/shared storage, so every output tensor is
    `.clone().contiguous()`; I carried over the source header metadata.
- **Anything in the task text or documentation that was unclear:** the task says
  "verify ... every tensor outside the 64 MLP tensors is identical in all three"
  without saying whether shape/dtype mismatch or value mismatch should be
  distinguished; I treat both as an abort. The 64 MLP names are fully specified,
  so I derive them from the layer pattern rather than diffing the checkpoints,
  which also makes the "exactly 64 merged" check meaningful.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- **Approximate time spent, if you can tell:** a few minutes; one script, one run.
