# T1 participant self-report

- Final artifact path: `out/T1/solution.py` (invoked via `out/T1/run.sh`), output at `out/T1/model.safetensors`.
- Number of times you executed the script or plan: 2 (one to produce the output, one re-run through `run.sh` to confirm it reproduces cleanly from a clean `out/T1/`).
- Which executions failed, and why (one line each): none failed.
- Pitfalls or surprises you hit (one line each):
  - The old-index -> new-index renumbering (0,1,3,4,6,7,9,10,11 -> 0..8) overlaps in range with the old indices, so an in-place rename (e.g. iterating and reassigning keys in one dict) risks a surviving block overwriting another before it's read; building a brand-new output dict and only ever writing each new key once sidesteps this rather than requiring a careful processing order.
  - `h.<i>.attn.bias` is a non-trainable causal-mask buffer, not a projection weight — it still has to move with its block during renumbering, and a regex matching only weight/bias parameter names could miss it.
- Anything in the task text or documentation that was unclear: no, the exact remapping (old -> new) was given explicitly, so there was no ambiguity to resolve.
- Tools used (condition F): `safetensors` 0.5.3 (`load_file`/`save_file`) for checkpoint I/O, plus Python's standard `re` module for key parsing — no merge-config tool was used. `mergekit`'s layer-slicing (passthrough) YAML config only expresses contiguous slice ranges; this task drops a non-contiguous set of blocks (2, 5, 8) and keeps the rest, which doesn't map cleanly onto that DSL. `torch-state-bridge`'s regex rewriting was also a candidate, but a ~50-line script gave me full control over collision detection and the exact required checks (drop verification, block count, total tensor count) without learning a second tool's config format for a one-off rename.
- Approximate time spent, if you can tell: a few minutes of design + implementation, plus verification against the input checkpoint (bit-exact equality check on all surviving tensors).
