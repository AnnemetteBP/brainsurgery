# T4 — Participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the first execution succeeded.
- **Pitfalls or surprises you hit:**
  - The base is sharded, the fine-tunes are single files, so the three loaders are not symmetric; the base has to be reassembled from `model.safetensors.index.json` before any comparison.
  - The ordering hazard is easy to avoid by building a fresh output dict rather than mutating the base dict in place — both task vectors are then necessarily taken against the pristine base.
  - `save_file` rejects non-contiguous/shared storage, so every output tensor is `.contiguous()` (and unchanged ones are cloned out of the mapped shard).
- **Anything in the task text or documentation that was unclear:** nothing material. The output dtype is only implied ("computed in float32", inputs are float32); I compute in float32 and write back the input dtype, which is float32 either way.
- **Tools used (condition F):**
  - `torch` 2.14.0 — tensor arithmetic and the bit-exact `torch.equal` comparison.
  - `safetensors` 0.5.3 — sharded and flat checkpoint load, single-file save.
  - I deliberately did **not** use `mergekit` 0.1.4 despite its `task_arithmetic` method: the task's step-1 precondition (every one of the 66 non-MLP tensors bit-identical across all three checkpoints, abort otherwise) and the "exactly 48 merged / exactly 114 written" counts are not expressible in a mergekit YAML config, and mergekit would additionally merge or copy tensors according to its own base-model logic rather than restricting the change to the 48 MLP tensors. A ~90-line script does the whole job with the required checks as hard failures.
- **Approximate time spent:** ~5 minutes.

## Checks enforced by the run (all raise `CheckFailed` -> exit code 1)

1. Identical tensor name sets across base/ft1/ft2, and 114 names total.
2. Exactly 48 names match the MLP pattern `model.layers.<i>.mlp.(gate|up|down)_proj.weight`.
3. Matching shape and dtype for every tensor across the three checkpoints.
4. Every one of the 66 non-MLP tensors bit-identical (`torch.equal`) in all three — checked before anything is written.
5. Exactly 48 tensors merged.
6. Output dict has 114 tensors, and the file re-read after writing has 114 tensors with the base key set.

Run output:

```
verified: 114 tensors, 66 shared tensors bit-identical
wrote out/T4/model.safetensors : 114 tensors, 48 merged, lambda=0.4
```
