# T4 — Participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the first execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The base is sharded across two safetensors files while the fine-tunes are single files, so the loader had to read `model.safetensors.index.json` and map each name to its shard rather than assuming one file per checkpoint.
  - The ordering hazard is easy to avoid only if the base tensor is read once per name and both task vectors are derived from that same unmodified tensor; I never write back into the base view.
  - `inputs/` also contains a `lora/` directory that belongs to a different task; matching MLP tensors by an anchored regex on `model.layers.<i>.mlp.{gate,up,down}_proj.weight` (48 matches, asserted) avoids any accidental overreach.
- **Anything in the task text or documentation that was unclear:** nothing material. The spec is explicit about lambda, the tensor set, dtype and the output path. It does not say whether `out/T4/` may hold config/tokenizer files as well; I wrote only `model.safetensors` plus my own script and this report, since grading only names the tensor file.
- **Tools used (condition F):**
  - `safetensors` 0.5.3 — lazy per-tensor reads via `safe_open` (so the three ~5 GB checkpoints are never fully resident at once) and `save_file` for the output.
  - `torch` 2.14.0 — float32 tensor arithmetic and `torch.equal` for the bit-exact backbone comparison.
  - **Why not `mergekit` 0.1.4**, the route the condition sketch suggests: its `task_arithmetic` method applies the scaled task vectors to *every* tensor in the model, so it cannot express "touch only these 48 MLP tensors and copy the other 66 through bit-exactly", and it has no hook to assert the frozen-backbone precondition or the 48/114 tensor counts before writing. Enforcing the required checks inside mergekit would have meant a post-hoc verification pass over its output anyway, so a ~120-line script on top of safetensors was both shorter and the only version where the checks actually gate the write. `transformers`/`peft` were not applicable (no architecture instantiation or adapters needed here).
- **Approximate time spent:** ~5 minutes total; the run itself takes 13 s wall clock.

## How the required checks are enforced

All checks raise a `Check` exception that exits non-zero, and all of them run *before* the output file is written:

1. **Shared-tensor verification** — the three key sets must be identical (and 114 names); every tensor must agree in shape and dtype across the three; each of the 66 non-MLP tensors must satisfy `torch.equal(base, ft1)` and `torch.equal(base, ft2)`.
2. **Exactly 48 merged** — the anchored MLP regex must match exactly 48 names, and the merge loop's own counter must equal 48.
3. **Exactly 114 output tensors** — asserted on the in-memory dict, and re-asserted by reopening the written file (which is saved to a `.tmp` path and renamed, so a failure cannot leave a truncated `model.safetensors` behind).

Independent spot-check after the run: all 66 non-MLP tensors are bit-identical to the base, and all 48 merged tensors match `base + 0.4*(ft1-base) + 0.4*(ft2-base)` to a relative Frobenius error below 1e-6.
