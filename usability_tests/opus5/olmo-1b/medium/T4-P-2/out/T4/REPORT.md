# T4 participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`, 114 tensors)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The base is sharded, so tensor lookup had to go through `model.safetensors.index.json` while `ft1`/`ft2` are single files; I wrote one small reader class that handles both instead of special-casing.
  - `inputs/` also contains a `lora/` directory that this task does not use; ignoring it was the right call but it is an easy thing to grab by mistake.
  - The stated ordering hazard is real: both task vectors have to be formed against the pristine `base[X]`, so I read `base` fresh per tensor and never wrote a partially merged value back into the base view.
  - The output is a single ~5 GB file, so I streamed tensors per name via `safe_open` rather than loading three whole checkpoints (that would have been ~15 GB resident).
  - `save_file` rejects non-contiguous tensors, so merged results are forced contiguous before saving.
- **Anything in the task text or documentation that was unclear:**
  - "identical" for the 66 shared tensors was not defined; I used bit-exact `torch.equal`, which matches the grader's bit-exact requirement for those tensors.
  - The task guarantees float32 inputs, so the "computed in float32" requirement was satisfied by asserting the dtype and doing the arithmetic in float32 rather than by an upcast.
- **Tools used (condition F):** n/a — condition P (Python + `torch` 2.14.0, `safetensors` 0.5.3).
- **Approximate time spent, if you can tell:** ~5 minutes.
