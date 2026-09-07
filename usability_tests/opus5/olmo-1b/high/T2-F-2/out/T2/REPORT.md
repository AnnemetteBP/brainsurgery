# T2 run record — participant self-report

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution passed all checks and wrote the output.
- **Pitfalls or surprises you hit (one line each):**
  - The route suggested for this task in `F-allowed.md` (transformers `prune_heads`) does not exist for this architecture: `OlmoModel` has no `_prune_heads` in transformers 5.12.1, so `prune_heads` would have raised rather than pruned.
  - Even had it worked, `prune_heads` + `save_pretrained` is the wrong shape of tool here: it records `pruned_heads` in the config and controls sharding itself, whereas the task wants one file with the original 114 names.
  - Directional asymmetry is the real trap: q/k/v are pruned on rows (dim 0) but o_proj on columns (dim 1), because o_proj *consumes* the head outputs; slicing o_proj on rows would have produced a checkpoint that still loads.
  - Input is sharded across two files, so the tensor set has to be reassembled from `model.safetensors.index.json` before anything else, and written back as a single file.
  - `torch.cat` of two slices is contiguous, but I asserted contiguity anyway since safetensors rejects non-contiguous tensors at save time.
- **Anything in the task text or documentation that was unclear:**
  - The task says the result "must be loadable as the same architecture with 15 heads per layer", but the required output is only `model.safetensors`; it is not stated whether a `config.json` with `num_attention_heads: 15` should be emitted alongside. Since grading compares tensors (key set, shapes, dtypes, values) and requirement 6 asks for a single file, I wrote only the checkpoint.
  - OLMo-1B-0724 has no per-layer norm parameters at all (114 = 16 layers x 7 tensors + embed + lm_head), so "every other tensor is unchanged" was unambiguous; on an architecture with `q_norm`/`k_norm` it would not have been.
- **Tools used (condition F):**
  - `torch` 2.14.0+cu130 — tensor slicing and `torch.cat`, plus `torch.equal` for the bit-exactness checks.
  - `safetensors` 0.5.3 — `safe_open` to stream both input shards and to read the output back, `save_file` to write the single output file.
  - `transformers` 5.12.1 — only to check whether the `prune_heads` route was viable (it is not, see above); not used in the solution.
  - Rationale: the operation is a deterministic, bit-exact slice of 64 named tensors. Every higher-level tool on the allowed list (mergekit, torch-state-bridge, peft) works at the granularity of whole tensors or key names, not sub-tensor blocks, so none of them can express "drop rows 640..767". A plain script over safetensors was both the shortest and the only route that guarantees bit-exactness.
- **Approximate time spent, if you can tell:** ~5 minutes, most of it the two ~5 GB read/write passes.

## Checks enforced by the run

The script raises `AssertionError` and writes nothing if any of these fails:

- input contains exactly 114 tensors, and the index and shards agree with no duplicates;
- every one of the 64 attention projections is `[2048, 2048]` before slicing;
- the four required layer-0 shape checks (`q/k/v -> [1920, 2048]`, `o -> [2048, 1920]`);
- the same shape check on all 16 layers, plus bit-exact equality against
  `index_select` of the kept indices (rows for q/k/v, columns for o_proj);
- output key set equals the input key set, exactly 114 tensors, dtypes preserved, all contiguous;
- every non-attention tensor equal to its input, same shape;
- after writing, a readback of the file on disk re-checks count, key set and every value.
