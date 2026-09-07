# T2 participant self-report

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none; the first execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - `transformers.prune_heads` was the route the condition suggests, but it rewrites the config and reindexes/renames nothing predictably for a bit-exact comparison; a direct slice keeps values bit-identical, so I went with a plain script.
  - The two axes differ (q/k/v are row blocks, o_proj is a column block); mixing them up is the obvious failure and is caught by the per-tensor shape asserts.
  - Input is sharded across two files, so the state dict has to be assembled from the index before slicing, and the output is a single unsharded file.
  - OLMo-1B-0724 has non-parametric layer norms, so 114 tensors = 16 layers x 7 + embeddings + lm_head; nothing else is head-bearing.
- **Anything in the task text or documentation that was unclear:** whether any non-tensor files (a `config.json` with `num_attention_heads: 15`) should accompany the output. The task names exactly one output file and grading compares tensors, so I wrote only `model.safetensors`.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — shard loading and single-file saving; the task is defined directly in terms of safetensors tensors.
  - `torch` 2.14.0 — `index_select` for the slice and `torch.equal` for the verification pass.
  - Deliberately not used: `transformers` `prune_heads` (mutates config and is not bit-exactness-oriented), `mergekit` (no per-head slicing primitive), `torch-state-bridge` (renames keys; names do not change here).
- **Approximate time spent:** ~5 minutes.

## Checks enforced by the run

The script aborts before writing if any assert fails. Beyond the required checks
(the four layer-0 projection shapes and the 114-tensor count), it also verifies:
the same shapes on all 16 layers; that exactly 64 tensors were sliced; that the
name set and every dtype are unchanged; that untouched tensors are element-wise
identical to the input; that the kept blocks line up with the original rows/columns
either side of the removed head; and it re-reads the written file and compares it
tensor-by-tensor.
