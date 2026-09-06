# T1 self-report (condition F, Pythia-1B)

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - Collision hazard avoided by building a fresh dict from a computed old->new map instead of renaming in place.
- Anything in the task text or documentation that was unclear: nothing
- Tools used (condition F): name, version, and why:
  - safetensors 0.5.3: load_file/save_file for checkpoint I/O.
  - torch 2.14.0: tensor container (used implicitly by safetensors; also for a separate bit-exact verification pass).
  - Chose a plain script over mergekit/torch-state-bridge: the task is a pure key remap, and a 40-line script keeps the required checks explicit and fails before writing.
- Approximate time spent, if you can tell: about 2 minutes
