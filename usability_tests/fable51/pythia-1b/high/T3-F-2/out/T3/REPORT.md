# Participant self-report: T3 (Pythia-1B, condition F, repeat 2)

- Final artifact path: `out/T3/solution.py` (run with `python out/T3/solution.py` from the sandbox root); output shards `out/T3/model-0000N-of-00009.safetensors` and `out/T3/model.safetensors.index.json`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - The task text quotes the embeddings as 206 MB each, which is their float16 size; after the required upcast they are 412 MB, so they still exceed the 256 MiB budget and sit alone in their own shards, but the number is stale.
  - `.*weight` would overreach onto embeddings and layer norms, so I built the 64 target names and the 48 buffer names explicitly from templates and asserted both sets exist in the input; no regex was used.
  - The mask buffers `attention.bias` are siblings of the real `attention.dense.bias` / `query_key_value.bias` parameters, so suffix matching must be exact, not substring-based.
- Anything in the task text or documentation that was unclear: whether the grader requires a specific shard assignment or shard file names; I used HuggingFace-style `model-XXXXX-of-XXXXX.safetensors` names and a greedy fill in the input's key order (alphabetical, as safetensors stores them), which satisfies the stated rules.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0: dtype casting (`tensor.to(torch.bfloat16)` for RNE) and byte-size accounting.
  - `safetensors` 0.5.3: `safe_open` for lazy reading and `save_file` for the shards.
  - I did not use `transformers.save_pretrained` because it applies one dtype to the whole model, keeps the buffers, and its shard budget counts differently; a 100-line script gave exact control over the mixed precision and the buffer drop.
- Approximate time spent, if you can tell: about 3 minutes including an independent re-verification (reload every shard, compare bit-exact against a fresh cast of the input, check index consistency and shard budgets).
