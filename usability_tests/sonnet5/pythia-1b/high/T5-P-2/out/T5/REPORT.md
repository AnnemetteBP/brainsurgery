## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `KeyError` — the regex that strips `.lora_A.weight` off the
    adapter tensor name left the captured base module name without its
    trailing `.weight`, so it didn't match any base tensor key; fixed by
    appending `.weight` when building the base name.
- Pitfalls or surprises you hit (one line each):
  - The adapter tensor names are prefixed with `base_model.model.`, which has
    to be stripped in addition to swapping `lora_A`/`lora_B` for the plain
    module name before looking the tensor up in the base checkpoint.
  - Needed to `.contiguous()` tensors before `save_file`, otherwise
    safetensors can reject non-contiguous views (didn't actually hit this
    since all tensors here were already contiguous, but added defensively).
- Anything in the task text or documentation that was unclear:
  - None; the task text gave the exact naming scheme, scale formula, and
    shard budget, which matched the checkpoint on inspection.
- Tools used (condition F): n/a (condition P)
- Approximate time spent, if you can tell: ~10 minutes
