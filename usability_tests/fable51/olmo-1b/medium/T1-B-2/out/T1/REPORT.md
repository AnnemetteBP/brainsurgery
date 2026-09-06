# T1 self-report (condition B, OLMo-1B-0724-hf)

- Final artifact path: `out/T1/model.safetensors` (plan: `out/T1/plan.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded (exit 0, 86 tensors written).
- Pitfalls or surprises you hit (one line each):
  - The collision hazard was avoided by ordering the `move` transforms by ascending old index after the `delete`, so every destination index is lower than its source and already vacant; `move` refuses existing destinations anyway, which would have caught a wrong order loudly.
  - Regex references are full-match, so `model\.layers\.1\..+` cannot accidentally match layers 10..15; I still anchored every pattern with the trailing `\.`.
- Anything in the task text or documentation that was unclear:
  - The README does not say explicitly that a sharded HF model directory can be given as an input path; the log confirmed the index was detected, so it worked without extra options.
  - The task's first required check names blocks 12..15 (post-renumber indices), while the removed blocks are 2, 6, 10, 14; I implemented the check as written (no `model.layers.12..15` remain) and added counts for 12 q_proj tensors, 84 block tensors and 86 total.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 2 minutes (doc reading, one plan write, one run).
