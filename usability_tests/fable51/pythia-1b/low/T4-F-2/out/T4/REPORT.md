# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (output `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - none; verification of the 180 shared tensors passed on the first run
- Anything in the task text or documentation that was unclear: nothing
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: float32 arithmetic, `torch.equal` for bit-exact shared-tensor checks
  - `safetensors` 0.5.3: `load_file` / `save_file`
  - mergekit was not used: its task-arithmetic path does not verify the shared-tensor precondition and would round-trip the untouched tensors through its own writer, so a 45-line script was simpler and safer
- Approximate time spent, if you can tell: about 2 minutes
