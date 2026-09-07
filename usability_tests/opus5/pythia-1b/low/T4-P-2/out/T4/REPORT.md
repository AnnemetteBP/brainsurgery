# T4 participant self-report (condition P)

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The ordering hazard is real but easy to avoid: both task vectors must be
    subtracted from the original `base[X]`, so I computed `b32` once per tensor
    and never wrote back into the base dict.
  - `inputs/` also contains a `lora/` directory that is irrelevant to this task;
    I ignored it.
  - Safetensors rejects non-contiguous / shared storage, so I called
    `.contiguous()` (and `.clone()` on pass-through tensors) before saving.
  - The equality check for the 180 untouched tensors is bit-exact
    (`torch.equal`), which is what the grader also does; no tolerance needed there.
- Anything in the task text or documentation that was unclear:
  - The task says "cast back to float16 (the base dtype)"; I cast to each
    tensor's own base dtype instead of hardcoding float16, which is equivalent
    here since all inputs are float16.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: a few minutes; one read of the
  inputs, one script, one run.
