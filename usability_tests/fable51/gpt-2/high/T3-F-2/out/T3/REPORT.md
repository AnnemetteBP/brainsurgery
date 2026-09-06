# T3 self-report (GPT-2 124M, condition F, repeat 2)

- Final artifact path: `out/T3/solution.py` (output shards + `model.safetensors.index.json` in `out/T3/`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The safetensors header stores keys in non-sorted order; I shard in sorted key order (what `safe_open(...).keys()` returns) so the layout is deterministic.
  - `h.<i>.attn.bias` looks like a parameter bias by name but is the causal-mask buffer; matched it with an anchored regex so the real `c_attn.bias` / `c_proj.bias` are kept.
  - Used anchored, explicit per-projection regexes instead of `.*weight` to avoid casting `wte`, `wpe` and layer-norm weights.
  - `wte.weight` (154 MB) exceeds the 64 MiB shard budget, so the greedy sharder flushes the current shard and stores it alone.
- Anything in the task text or documentation that was unclear:
  - The task does not specify shard file names or the order tensors are packed into shards; I used the HuggingFace convention `model-0000i-of-0000n.safetensors` and greedy packing in sorted key order. Grading says "sharding rules", so I assume file names are not compared.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: dtype cast (`tensor.to(torch.bfloat16)`, RNE) and `torch.equal` for bit-exact passthrough checks.
  - `safetensors` 0.5.3: `safe_open` to read, `save_file` to write each shard with `{"format": "pt"}` metadata.
  - Plain Python `re`/`json` for name matching and the index file.
  - Did not use `transformers.save_pretrained`: it applies one dtype to the whole model and would still require hand-building the mixed-precision state dict, and it would drop/keep buffers according to the module, not the spec. `mergekit` dtype conversion is likewise whole-checkpoint. A short script was simpler and fully controllable.
- Approximate time spent, if you can tell: about 3 minutes (inspect inputs, write script, run once, verify independently).

Result: 4 shards (59 / 58 / 30 / 1 tensors; 66,259,968 / 66,256,896 / 40,983,552 / 154,389,504 bytes of tensor data), 148 tensors, 48 bfloat16, 100 float32, 12 buffers dropped. An independent re-read of the shards confirmed every bf16 tensor equals `src.to(bfloat16)` and every float32 tensor is bit-identical to the input.
