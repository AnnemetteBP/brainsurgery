# T4 (condition P) — participant self-report

- Final artifact path: `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The base is sharded, so it has to be reassembled from `model.safetensors.index.json` before any comparison; the fine-tunes are single files.
  - The ordering hazard: both task vectors must be taken against the original base, so I built the output into a separate dict and never read back a merged tensor.
  - `save_file` rejects non-contiguous/shared storage, so every tensor is `.contiguous()` before writing.
- Anything in the task text or documentation that was unclear:
  - "computed in float32" is trivially satisfied here since all inputs are already float32; I still cast explicitly so the code would stay correct for a lower-precision checkpoint.
  - The task does not say whether the output should also carry `config.json`/tokenizer files; I wrote only `model.safetensors` as literally specified.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 10 minutes.
