# T1 self-report (condition B)

- Final artifact path: `out/T1/plan.yaml` -> `out/T1/model.safetensors`
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: crash (PlanLoaderError) — `assert.exists` takes a bare tensor-ref payload, not a mapping with `of:`; I had written `not: { exists: { of: ... } }`.
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard: solved by deleting blocks 2/6/10/14 first and then moving surviving blocks in ascending old-index order, so every destination index is already free.
  - `exists` payload shape differs from `count`/`iszero` (which use `of:`); the help text made this clear only on re-reading.
  - Dots in tensor names must be escaped in `from`/`target` regexes to avoid overmatching (e.g. layer `1` vs `11` is handled by the trailing `\.`).
- Anything in the task text or documentation that was unclear:
  - The task says the output is "a single file `out/T1/model.safetensors`" while grading points at the directory `out/T1`; I wrote the single file path explicitly with `format: safetensors` to avoid sharding.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes.

## Plan sketch

1. `assert count(.*) == 114` on the input.
2. `delete model\.layers\.(2|6|10|14)\..*`, then `assert count(.*) == 86`.
3. Ten `move` transforms in ascending old index: 3→2, 4→3, 5→4, 7→5, 8→6, 9→7, 11→8, 12→9, 13→10, 15→11, using capture `\1` for the tensor suffix.
4. Final `assert all`: no `model.layers.(12|13|14|15).*` exists; exactly 12 `self_attn.q_proj.weight`; 84 block tensors; 86 tensors total.

Verified after the run: 86 tensors, block indices 0..11 contiguous, the 2 non-block tensors untouched, spot-checked bit-exact equality for old 0/3/9/15 → new 0/2/7/11, dtype float32.
