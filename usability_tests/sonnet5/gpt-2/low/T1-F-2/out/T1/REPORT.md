## Participant self-report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none — block renumbering by
  precomputing an old-index-to-new-index map before writing avoids the
  collision hazard entirely (survivors are relabeled into a fresh dict, never
  overwriting a not-yet-processed original key).
- Anything in the task text or documentation that was unclear: none
- Tools used (condition F): plain script on top of `safetensors` 0.5.3
  (`safetensors.torch.load_file`/`save_file`) only. Chose this over
  mergekit's layer-slicing passthrough or torch-state-bridge's regex
  rewriting because the rename is a simple, fully-specified bijection on 160
  known tensor names — a ~15-line script with inline checks is easier to
  verify by inspection than translating the mapping into another tool's
  config format, and it lets the required checks (no blocks 9/10/11, exactly
  9 surviving blocks, exactly 121 tensors) be asserted directly before any
  write, with a guaranteed non-zero exit and no partial output on failure.
- Approximate time spent, if you can tell: a few minutes
