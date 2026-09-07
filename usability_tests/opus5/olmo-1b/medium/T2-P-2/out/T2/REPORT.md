# T2 self-report (condition P)

- Final artifact path: `out/T2/model.safetensors` (script: `out/T2/solution.py`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Input is sharded, so the state dict had to be reassembled from
    `model.safetensors.index.json` before slicing; output is a single file.
  - q/k/v are separate (not fused) in OLMo-1B-0724-hf, so head 5 is a plain row
    block `640:768` in each of the three; only `o_proj` is the column-block case.
  - `safetensors.save_file` rejects tensors sharing storage, and slices of an
    mmapped shard are views, so every tensor is `.contiguous().clone()`d before
    saving (OLMo-1B is also untied, but the clone makes that moot).
- Anything in the task text or documentation that was unclear: nothing; the task
  gave the exact row/column ranges and the target shapes, so the only judgement
  call was how to read the sharded input and how to write one output file.
- Tools used (condition F): n/a (condition P: torch 2.14.0 + safetensors 0.5.3).
- Approximate time spent, if you can tell: a few minutes; one read of the index
  file, one script, one run (the run itself is dominated by moving ~4.8 GB).
