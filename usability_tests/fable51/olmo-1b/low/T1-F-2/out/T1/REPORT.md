# T1 self-report

- Final artifact path: `out/T1/solution.py` (output `out/T1/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each): none; building a fresh output dict (rather than renaming in place) removes the renumbering-collision hazard entirely, and an explicit collision check guards it anyway.
- Anything in the task text or documentation that was unclear: the "Required checks" say no tensor of blocks 12-15 may remain, which is a necessary but weak check; I additionally verify the block set is exactly 0..11 and that every surviving tensor is bit-identical to its source.
- Tools used (condition F): safetensors 0.5.3 (safe_open / save_file for sharded load and single-file save), torch 2.14.0 (tensor equality), Python stdlib re/json. Chose a plain script because the task is a pure key rename with a fixed mapping; mergekit or torch-state-bridge would add config surface without adding safety over a 60-line script with explicit checks.
- Approximate time spent, if you can tell: about 2 minutes
