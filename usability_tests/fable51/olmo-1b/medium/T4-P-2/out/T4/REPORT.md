# Participant self-report: T4 (condition P, OLMo-1B-0724-hf)

- Final artifact path: `out/T4/solution.py` (output checkpoint `out/T4/model.safetensors`, 114 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base is sharded (index json + two shards) while ft1/ft2 are single files, so the loader had to handle both layouts.
  - Task vectors must both be taken against the untouched base tensor; the script computes `ft1 - base` and `ft2 - base` before adding anything, and never mutates `base`.
  - Three float32 1B checkpoints plus the output are about 20 GB resident; fine on this machine but worth streaming per tensor on smaller boxes.
- Anything in the task text or documentation that was unclear: nothing; the MLP tensor set, lambda and check list were explicit.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3).
- Approximate time spent, if you can tell: about 3 minutes.
