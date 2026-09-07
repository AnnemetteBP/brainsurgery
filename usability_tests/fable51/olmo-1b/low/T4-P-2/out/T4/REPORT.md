# Participant self-report: T4 (condition P)

- Final artifact path: `out/T4/solution.py` (output `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - The base is sharded; needed to load both shards via the index and check the union matches the weight_map.
  - Task vectors are computed against the untouched base dict; the output is built in a separate dict so ordering cannot corrupt the second task vector.
  - An `inputs/lora/` directory is present but is not part of T4; ignored.
- Anything in the task text or documentation that was unclear: nothing; the formula and checks were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
