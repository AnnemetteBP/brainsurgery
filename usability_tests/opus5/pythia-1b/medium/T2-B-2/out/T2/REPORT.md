# T2 self-report (condition B: BrainSurgery plan)

- Final artifact path: `out/T2/plan.yaml` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so
    the per-layer edits cannot be written once with a regex over layers; the plan
    is generated as 16 explicit layer blocks (144 transforms).
  - Tensor references are full-match regexes, so every `.` in a tensor name has to
    be escaped (`gpt_neox\.layers\.0\....`) to avoid matching a sibling name.
  - `copy`/`concat` destinations must not already exist, so the rebuilt tensor is
    written to a temporary name, the original is `delete`d, and the temp is `move`d
    back to the original name (`move` refuses to overwrite too).
  - The GPT-NeoX fused QKV is interleaved per head, so head 5 is one contiguous
    768-row block (3840..4607) rather than three separate q/k/v strips — the cut is
    a single two-piece concat on dim 0.
  - `attention.dense.weight` is `[out, in]`, so heads are column blocks and the
    concat is on dim 1 (`[:, :1280]` + `[:, 1536:]`), not dim 0.
- Anything in the task text or documentation that was unclear: nothing blocking; the
  task text spelled out the exact row/column ranges to keep, which removed the layout
  guesswork. The README does not say explicitly that `concat` sources must be
  single-tensor refs — that is only in `help: concat`.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: ~10 minutes.

## What the plan does

Per layer `i` in 0..15:
1. `concat` rows `[:3840]` + `[4608:]` of `attention.query_key_value.weight` (dim 0) into a temp.
2. `concat` the same ranges of `attention.query_key_value.bias` (dim 0) into a temp.
3. `concat` columns `[:, :1280]` + `[:, 1536:]` of `attention.dense.weight` (dim 1) into a temp.
4. `delete` the three originals, then `move` each temp back onto the original name.

Then the four required checks as `assert` transforms (three `shape`, one
`count: { of: '.*', is: 244 }`) before `output` writes
`out/T2/model.safetensors`.

## Verification

Output has 244 tensors; layer 0 shapes are `[5376, 2048]`, `[5376]`, `[2048, 1792]`;
dtypes stay float16. Independently re-derived the expected slices for all 16 layers
and confirmed bit-exact equality, and confirmed every untouched tensor is unchanged.
