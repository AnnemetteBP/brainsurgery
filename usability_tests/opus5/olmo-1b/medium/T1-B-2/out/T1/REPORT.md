# T1 self-report (condition B)

- **Final artifact path:** `out/T1/plan.yaml` -> `out/T1/model.safetensors` (86 tensors, float32)
- **Number of times you executed the script or plan:** 1 (plus one separate read-only
  verification plan run against the finished output; it wrote nothing)
- **Which executions failed, and why (one line each):** none, the single run passed.
- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collisions: `move` refuses an existing destination, so the ten renames had
    to be ordered by ascending target index (3->2, 4->3, ... 15->11) after deleting blocks
    2/6/10/14; that way each destination slot is always free and no block overwrites a survivor.
  - Blocks 0 and 1 keep their index, so they must simply not be moved (a self-move 0->0
    would fail on `dest_exists`).
  - Regex refs are full-match, so dots must be escaped (`model\.layers\.3\.(.*)`) to avoid
    `model.layers.13...` being caught by a `3` pattern; the delete pattern
    `model\.layers\.(2|6|10|14)\..*` is safe only because of the trailing escaped dot.
  - `count: { is: 0 }` felt risky for "nothing remains", so the absence checks use
    `not: { exists: ... }` instead.
  - `output.path` with a `.safetensors` suffix writes a single file; a suffix-less path
    would have sharded into a directory.
- **Anything in the task text or documentation that was unclear:**
  - The README documents that capture groups can be used in `to`/`right` only in the
    `assert.equal` section; that the same `\1` rewrite applies to `copy`/`move` is stated
    there but not in the `move` help text itself.
  - Nothing else; the required checks were unambiguous.
- **Tools used (condition F):** n/a (condition B: only `brainsurgery` and the doc pack).
- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading `help.txt`.

## What the plan does

1. Asserts the input is the 16-layer model (114 tensors, 16 `q_proj`).
2. `delete` every tensor of blocks 2, 6, 10, 14 in one pattern.
3. Ten `move` transforms renumber the survivors in ascending target order.
4. Final `assert: all` implements the required checks: no block 12-15 tensor remains,
   exactly 12 `q_proj` (so 12 blocks), 84 block tensors in indices 0..11, both non-block
   tensors present, and exactly 86 tensors total.

Independently verified afterwards with a read-only plan that loads `out/T1/model.safetensors`
alongside `inputs/base` and `assert: equal`s every renumbered block against its original
(0<-0, 1<-1, 2<-3, 3<-4, 4<-5, 5<-7, 6<-8, 7<-9, 8<-11, 9<-12, 10<-13, 11<-15) plus both
non-block tensors: all bit-exact, dtype float32.
