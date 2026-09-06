# T3 participant self-report

- Final artifact path: `out/T3/solution.py` (output: `out/T3/model-0000{1..4}-of-00004.safetensors` + `out/T3/model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none
- Pitfalls or surprises you hit (one line each):
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget, so the sharder must allow a single oversized tensor alone in a shard.
  - `h.<i>.attn.bias` is a mask buffer, not a parameter; matched it with an anchored regex so projection biases are untouched.
- Anything in the task text or documentation that was unclear: nothing significant; shard file naming is unspecified, so I used the HuggingFace `model-XXXXX-of-XXXXX.safetensors` convention.
- Tools used (condition F): torch 2.14.0 (dtype cast) and safetensors 0.5.3 (`load_file`/`save_file`), in a plain script. I chose a script over `transformers.save_pretrained` because it gives exact control over which tensors are cast, guarantees buffers are dropped, and lets me enforce the 64 MiB tensor-data budget directly.
- Approximate time spent, if you can tell: about 2 minutes.
