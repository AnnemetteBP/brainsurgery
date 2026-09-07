# Participant self-report

- Final artifact path: `out/T2/plan.yaml` (executed with `brainsurgery cli out/T2/plan.yaml`), output at `out/T2/model.safetensors`.
- Number of times you executed the script or plan: 1 (the final `out/T2/plan.yaml`). Before writing it I ran a few scratch probe plans outside `out/T2/` (in a now-deleted `out/_probe/` scratch dir) to confirm undocumented behavior of `concat`/`move`/`assert` on one layer; those don't count as attempts on the graded plan.
- Which executions failed, and why (one line each): none — the graded plan (`out/T2/plan.yaml`) succeeded on its first and only execution.
- Pitfalls or surprises you hit (one line each):
  - `concat`'s and `split`'s `from` items must each resolve to exactly one tensor (no regex fan-out across layers like `copy`/`move`/`delete` get), so the per-layer slice-and-concat had to be written out explicitly for all 16 layers rather than as one regex-driven block; I generated that repetitive block with a short local script rather than hand-typing it.
  - `concat` destinations must not already exist, so the pruned tensors had to land under temporary `..._pruned` names first, then the originals were `delete`d and the temporaries `move`d onto the original names (regex+capture-group `move`/`delete` worked fine across all layers for this part).
  - Output path matters: pointing `output.path` at a directory (or an extension-less path) makes brainsurgery write a sharded directory (`model-00001-of-0000N.safetensors` + index), not the single file the task requires; giving `output.path` a `.safetensors` filename directly produces the single file.
- Anything in the task text or documentation that was unclear: the docpack doesn't state outright whether `concat`/`split` source references support the same regex/capture multi-match expansion as `copy`/`move`/`delete`; I confirmed empirically (via `help.txt`'s comparative phrasing and a scratch test) that they don't, which is what drove the per-layer unrolling.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: roughly 20-25 minutes, including exploratory checks of the doc pack and a couple of scratch-directory probes before writing and running the final plan.
