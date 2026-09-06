# T4 self-report

- **Final artifact path:** `out/T4/solution.py` (runner: `out/T4/run.sh`); output `out/T4/model.safetensors`
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The ordering hazard is easy to avoid by never mutating `base`: both task vectors are computed from the loaded base dict, and results go into a separate `out` dict.
  - Wrote the MLP name set explicitly (12 layers x {c_fc, c_proj} x {weight, bias}) instead of a regex, so `mlp.c_proj` cannot be over- or under-matched.
  - safetensors needs contiguous tensors, so merged tensors are `.contiguous()` before saving.
- **Anything in the task text or documentation that was unclear:**
  - The task says compute in float32 and the inputs are already float32, so the explicit `.to(torch.float32)` / cast back is a no-op here; I kept it so the code is correct if a dtype ever differs.
  - "Identical" for the 112 shared tensors was taken as bit-exact (`torch.equal`), which matches the grading description.
- **Tools used (condition F):** `safetensors` 0.5.3 (load/save) and `torch` 2.14.0 (tensor arithmetic and `torch.equal` comparison). I did not use mergekit: its task-arithmetic merge operates on whole HF models with its own dtype/output conventions and would not by itself enforce the required checks (shared-tensor verification, exactly-48-merged, exactly-160-out). A ~60-line script does the arithmetic exactly as specified and fails loudly on each required check, both before writing and by re-reading the written file.
- **Approximate time spent:** a few minutes; one read of the inputs, one script, one run.
