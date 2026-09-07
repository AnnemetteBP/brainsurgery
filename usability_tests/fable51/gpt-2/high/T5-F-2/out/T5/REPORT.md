# T5 (GPT-2 124M), condition F: participant self-report

- Final artifact path: `out/T5/solution.py` (run as `python out/T5/solution.py` from the sandbox root). Output: `out/T5/model-0000{1..5}-of-00005.safetensors` plus `out/T5/model.safetensors.index.json`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution passed all required checks and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - The base checkpoint keys have no `transformer.` prefix and include the `attn.bias` causal-mask buffers, so the peft / transformers `merge_and_unload` + `save_pretrained` route would rename every key and drop the 12 mask buffers; that rules it out for an exact-key-set grader.
  - The adapter keys (`base_model.model.h.<i>...`) do not match what peft expects for an HF GPT2LMHeadModel (`base_model.model.transformer.h.<i>...`); `PeftModel.from_pretrained` on these inputs silently loads a zero adapter and warns about 24 missing keys, so a naive peft merge yields the unchanged base.
  - `fan_in_fan_out = true`: the delta `scale * (B @ A)` is `[out, in]` and had to be transposed to Conv1D `[in, out]` before adding; a shape check guards this.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget and must sit alone in its own shard; the greedy packer starts a new shard when the next tensor would overflow, and only allows an oversized shard if it holds exactly one tensor.
- Anything in the task text or documentation that was unclear: nothing material. The task lists `target_modules = ["attn.c_attn"]` while `adapter_config.json` says `["c_attn"]`; irrelevant here because I map names from the adapter tensor keys, not from the config.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: `safe_open` to read base and adapter, `save_file` per shard. Chosen because it preserves exact key names, dtypes and bytes.
  - `torch` 2.14.0: float32 matmul, scaling, transpose and add for the merge.
  - `peft` 0.20.0 and `transformers` 5.12.1: used only for an out-of-band cross-check after the run (adapter weights injected into the PeftModel in memory, then `merge_and_unload`); the merged `c_attn` weights matched my output with relative error 0.0. Not used in the solution itself for the reasons above.
  - Not used: mergekit (no LoRA merge that preserves this key set), torch-state-bridge (no key rewriting needed beyond a regex).
- Verification beyond the required checks: reloaded the shards via the index, confirmed the key set equals the base, all 148 non-adapted tensors are bit-exact (`torch.equal`), and the 12 merged weights have a relative Frobenius error of 2.6e-8 against a float64 recomputation.
- Approximate time spent, if you can tell: about 3 minutes wall clock, one script execution plus two verification snippets.
