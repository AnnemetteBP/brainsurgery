# T2 (Pythia-1B) — Participant self-report, condition B

- Final artifact path: `out/T2/plan.yaml` (output checkpoint `out/T2/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so the per-layer pruning cannot be written as a single regex-driven transform; the plan contains 48 explicit concats (3 per layer), generated with a shell loop rather than hand-typed.
  - `concat` writes only to a new name, so the pruned tensors go to a `pruned.` prefix, then a regex `delete` removes the 48 originals and a regex `move` with a `\1` capture renames the pruned tensors back to the original names in two transforms.
  - Slice syntax on references (`name::[:3840, :]`, `name::[:, 1536:]`) works inside `concat` sources, which is what made the head removal expressible without Python.
- Anything in the task text or documentation that was unclear:
  - The README does not say whether `concat` accepts a pattern `from`; the `help` text does ("must resolve to exactly one tensor"), so read `help.txt` for the exact constraint.
  - Nothing in the task text was unclear; the row/column ranges were given explicitly.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes.
