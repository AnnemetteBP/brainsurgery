# T1 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T1/plan.yaml` (plan), `out/T1/model.safetensors` (output, 86 tensors, 4,045,416,024 bytes)

- **Number of times you executed the script or plan:** 1 (`brainsurgery out/T1/plan.yaml`, exit 0)

- **Which executions failed, and why (one line each):**
  - None. The single execution of `out/T1/plan.yaml` succeeded and wrote the output.

- **Pitfalls or surprises you hit (one line each):**
  - Renumbering collision hazard: I ordered the moves by ascending *old* index (3->2, 4->3, 5->4, 7->5, 8->6, 9->7, 11->8, 12->9, 13->10, 15->11) so every destination slot is already vacated, either by the `delete` of blocks 2/6/10/14 or by the preceding move; descending order would have collided with `move`'s "destination must not exist" rule.
  - Digit-prefix overreach: `model.layers.1.` must not also catch `model.layers.11/12/13/15`. BrainSurgery regexes are full-match and I escaped every dot, so `model\.layers\.1\.(.*)` is unambiguous; the same holds for the delete alternation `(2|6|10|14)`.
  - I loaded the same input directory twice, as `model` (edited, written out) and `base` (pristine, read-only). Output-alias inference still resolved to `model` because `assert` does not count as a write; this let me prove each renumbered block bit-exact against its intended source block instead of only counting tensors.
  - Sharded input, single-file output: the input is two shards plus an index, but giving `output.path` a `.safetensors` suffix produced one flat file with no index, as the task requires.
  - I confirmed separately (on throwaway plans, not `out/T1/plan.yaml`) that a failing `count` or `not: exists` assert exits non-zero and writes no output file, so the required checks really do fail loudly.

- **Anything in the task text or documentation that was unclear:**
  - The `move` help documents only single-tensor examples and does not say whether regex capture-group rewriting (`\1`) is supported for destinations; I inferred it from the `copy` examples in `docpack/examples/` and from the `assert.equal` docs, which describe `right` as being resolved "like `to` in copy/move".
  - The `dtype` and `shape` assert help is phrased in the singular ("the tensor"), so it was not obvious that they also accept a multi-match pattern; the MoE example plan shows they do.
  - The README's documentation links point at an absolute path on another machine (`/Users/petersk/...`), so they are dead in the doc pack.

- **Tools used (condition F):** n/a (condition B)

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading `docpack/help.txt` and the example plans; the plan run itself took ~11 s.
