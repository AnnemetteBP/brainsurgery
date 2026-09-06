# Participant self-report: T3 (GPT-2 124M, condition P)

- Final artifact path: `out/T3/solution.py` (output: `out/T3/model-0000{1..4}-of-00004.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 succeeded.
- Pitfalls or surprises you hit (one line each):
  - The buffer `h.<i>.attn.bias` shares a suffix with real biases such as `h.<i>.attn.c_proj.bias`, so I used an explicit key set instead of a `bias` pattern.
  - A regex like `.*weight` would also hit `wte`, `wpe` and the layer norms, so the 48 cast targets are enumerated explicitly per layer with expected shapes asserted.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget and had to be placed alone in its own shard; the greedy packer handles that as a special case.
- Anything in the task text or documentation that was unclear:
  - Whether the reference expects a specific shard packing order or file naming; I used safetensors' sorted key order, greedy first-fit packing, and HF-style `model-XXXXX-of-XXXXX.safetensors` names, and assumed the grader checks the sharding rules rather than exact shard assignment.
- Tools used (condition F): n/a (condition P: torch 2.14.0, safetensors 0.5.3)
- Approximate time spent, if you can tell: about 2 minutes
