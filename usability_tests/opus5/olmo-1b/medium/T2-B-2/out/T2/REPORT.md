# T2 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution passed all asserts and wrote the output.
- **Pitfalls or surprises you hit (one line each):**
  - `concat` requires each `from` ref to resolve to exactly one tensor, so it cannot be batched with a regex across layers; the plan is generated with 64 explicit `concat` lines (16 layers x 4 projections).
  - `concat`/`copy` destinations must not already exist, so the pruned tensor is built under a `tmp.<i>.<p>_proj.weight` name, the originals are deleted with one regex `delete`, and one regex `move` with backreferences (`tmp\.(\d+)\.([qkvo])_proj\.weight` -> `model.layers.\1.self_attn.\2_proj.weight`) restores all 64 names at once.
  - Tensor refs are full-match regexes, so dots must be escaped in the `delete`/`move` patterns to avoid over-matching; the literal per-layer `concat` refs are unambiguous anyway.
  - The head axis differs per tensor: q/k/v are row blocks (`::[:640, :]` + `::[768:, :]`, `dim: 0`), o_proj is column blocks (`::[:, :640]` + `::[:, 768:]`, `dim: 1`).
  - Head 5 of 128 dims spans rows/cols 640..767, so the kept ranges are 0..639 and 768..2047.
- **Anything in the task text or documentation that was unclear:** nothing blocking. The README does not say explicitly that `concat` sources cannot be patterns; that only shows up in `help: concat`. It is also not stated whether `move` reorders the state dict (it appends), which would matter if key order were graded.
- **Tools used (condition F):** n/a (condition B).
- **Approximate time spent, if you can tell:** ~5 minutes, most of it reading `help.txt` for `concat`/`split`/`move` semantics.

## Approach

Per layer `i` and projection `p in {q,k,v}`:
`concat: { from: ['...<p>_proj.weight::[:640, :]', '...<p>_proj.weight::[768:, :]'], to: 'tmp.<i>.<p>_proj.weight', dim: 0 }`
and for `o_proj` the same with column slices and `dim: 1`. Then one `delete` of
`model\.layers\.\d+\.self_attn\.[qkvo]_proj\.weight`, one `move` restoring the original names,
then the five required asserts (four `shape`, one `count: {of: '.*', is: 114}`) before `output`.

## Verification

The run's own asserts passed. Independently, the output safetensors header shows 114 tensors,
with layer 0 and layer 15 q/k/v at `[1920, 2048]` and o_proj at `[2048, 1920]`, all `F32`.
