# T2 report

## Participant self-report

- Final artifact path: `out/T2/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` entry and the `to` destination to resolve to exactly one
    existing tensor, so it cannot batch over all 16 layers via a regex/capture pattern the
    way `copy`/`cast`/`assign` can; the plan needed one `concat` per (layer, projection)
    rather than a single pattern-based transform.
  - `concat` (and `copy`/`cast`) destinations must not already exist, so each pruned
    projection was built under a `tmp_...` name, the original deleted, then `move`d back
    onto the original tensor name, rather than overwriting in place.
  - Confirmed the output path needs an explicit `.safetensors` file suffix (not a bare
    directory) to get a single `out/T2/model.safetensors` file instead of sharded
    `model-0000N-of-0000N.safetensors` + index output.
- Anything in the task text or documentation that was unclear: none; the row/column block
  boundaries (`0..639`, `768..2047`) and the per-tensor axis (rows for q/k/v, columns for o)
  were given explicitly in TASK.md, and `docpack/help.txt`'s `concat` example
  (`concat: { from: ['a::x::[:, :4]', 'a::x::[:, 4:]'], to: a::x_rebuilt, dim: 1 }`) mapped
  directly onto the required slice-and-reassemble operation.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: one exploration pass through `docpack/README.md`
  and the relevant `help.txt` sections (`concat`, `copy`, `move`, `delete`, `assert`), then
  one plan write and one successful run; no iteration needed.
