# T1 (condition P) — participant self-report

- Final artifact path: `out/T1/model.safetensors` (script: `out/T1/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Input is sharded, so keys must come from `model.safetensors.index.json` and tensors be read per shard rather than from a single file.
  - Renumbering collision hazard: avoided entirely by building a fresh output dict keyed by the new names instead of renaming in place, plus an explicit `destination collision` guard.
  - Tensors read via `safe_open` are views into the mmapped shard; called `.contiguous()` before `save_file` to avoid shared-storage/save errors.
- Anything in the task text or documentation that was unclear:
  - The task says "a single file `out/T1/model.safetensors`" while grading points at the directory `out/T1`; I wrote only the safetensors file and did not copy the config/tokenizer, since the required result lists exactly 86 tensors and nothing else.
  - No `metadata` format was specified; I wrote `{"format": "pt"}`, the HF default.
- Tools used (condition F): n/a — condition P (plain PyTorch + safetensors).
- Approximate time spent, if you can tell: ~3 minutes.
