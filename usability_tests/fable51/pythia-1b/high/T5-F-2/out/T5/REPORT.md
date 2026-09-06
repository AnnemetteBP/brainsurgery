# T5 self-report (Pythia-1B, condition F)

- Final artifact path: `out/T5/solution.py` (run as `python out/T5/solution.py`
  from the sandbox root); output shards `out/T5/model-0000N-of-00006.safetensors`
  plus `out/T5/model.safetensors.index.json`.
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - None failed. Execution 1 produced a valid 4-shard checkpoint under the
    literal 512 MiB rule (each embedding at 206 MB fits with other tensors);
    I re-ran after adding an explicit rule that stores `gpt_neox.embed_in.weight`
    and `embed_out.weight` alone, as the task text says they are, giving 6 shards.
- Pitfalls or surprises you hit (one line each):
  - The task says the two embeddings are "larger than" the 512 MiB budget, but
    they are 206 MB each; I handled both readings by listing them as standalone.
  - The base checkpoint keeps non-parameter buffers (`attention.bias` uint8 mask,
    `attention.masked_bias`, `rotary_emb.inv_freq`) that transformers 5 would drop
    on load, so a peft `merge_and_unload` route could not reproduce the exact
    244-key set bit-exactly; a direct safetensors script was safer.
  - `adapter_config.json` lists `target_modules = ["query_key_value"]` rather than
    the `attention.query_key_value` in the task text; the adapter tensor names
    were parsed directly, so this did not matter.
- Anything in the task text or documentation that was unclear:
  - The inconsistency between the 512 MiB budget and the "stored alone" claim
    for the 206 MB embeddings (see above).
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: read the base and adapter, write shards with `save_file`.
  - `torch` 2.14.0: float32 matmul for `B @ A`, scaling, add, cast to float16.
  - Python stdlib `json`, `re`, `os`: adapter config, name mapping, index file.
  - Not used: `peft`/`transformers` (model instantiation drops buffers and
    would not preserve the exact key set), `mergekit` (no LoRA merge for this
    checkpoint layout without a model).
- Approximate time spent, if you can tell: about 5 minutes.

Verification performed after the run (separate check, not part of solution.py):
all 244 keys, shapes and dtypes match the base; 228 unchanged tensors are
bit-exact; the 16 merged weights have relative Frobenius error at most 2.1e-4
against a float32 recomputation; index `total_size` matches the tensor bytes.
