# T4 (OLMo-1B-0724-hf), condition F: participant self-report

- Final artifact path: `out/T4/solution.py` (writes `out/T4/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all checks.
- Pitfalls or surprises you hit (one line each):
  - Base is sharded (two files + index) while ft1/ft2 are single files, so the reader had to map keys to shards and cross-check the index against shard contents.
  - Three float32 checkpoints are ~5 GB each; loaded lazily via `safe_open` per tensor so only the output dict is held in memory.
  - `torch.equal` treats NaN != NaN, so the shared-tensor comparison views float32 as int32 to be truly bit-exact.
- Anything in the task text or documentation that was unclear: nothing material. Whether "identical" meant bit-exact or allclose was implicit; I used bit-exact, which the grading section confirms.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: float32 arithmetic for the task-vector merge and tensor comparison.
  - `safetensors` 0.5.3: `safe_open` for lazy sharded/single-file reading, `save_file` for the single output file.
  - Not used: `mergekit` 0.1.4 task arithmetic. It does not perform the required shared-tensor precondition check or the "exactly 48 merged" check, applies the arithmetic to every tensor (risking non-bit-exact copies, e.g. -0.0 -> +0.0), and writes a sharded HF directory rather than one file. A ~120-line script gave direct control over all three required checks.
- Approximate time spent, if you can tell: about 5 minutes wall clock, of which ~14 s was the run itself.
