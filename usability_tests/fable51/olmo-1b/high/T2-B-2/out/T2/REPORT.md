# Participant self-report: T2 (OLMo-1B-0724-hf head pruning), condition B

- Final artifact path: `out/T2/plan.yaml` (output checkpoint `out/T2/model.safetensors`, 114 tensors, single file)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: failed on an extra check I added after the required ones, `assert: { count: { of: 'tmp\..*', is: 0 } }`; `count` raises "matched zero tensors" instead of counting 0, so a zero-count check must be written as `not: { exists: ... }`. The surgery itself and all required checks had already passed; the failure happened before the output was written, so no partial output.
  - Execution 2: success.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source reference to resolve to exactly one tensor, so it cannot be applied pattern-wise across 16 layers; I avoided 64 explicit concat blocks by using `fill` (sliced source gives the pruned shape) + two sliced `assign` steps + `delete` + `move`, all regex-based.
  - `count ... is: 0` cannot express "no matches" (see execution 1).
  - Approach relies on `fill` accepting a sliced regex source with capture rewrite in `to`; it does, but the README/help only shows this for `copy`/`assign`.
- Anything in the task text or documentation that was unclear:
  - The README does not say that zero matches are an error for `count`/`shape`/`dtype` style references; discovered from the traceback.
  - Whether `fill` participates in capture-based destination synthesis is not stated (the interfaces reference lists "including copy, move, assign, add_, subtract_").
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 5 minutes

Additional verification (not counted as a plan execution): `out/T2/verify/check.yaml`, a check-only plan with no
output, loads `inputs/base` and `out/T2/model.safetensors` and asserts: 114 tensors, all float32, pruned shapes on
all 16 layers, every non-projection tensor bit-equal to the input, and each pruned q/k/v/o tensor piece-wise
bit-equal to the kept input slices (rows/cols 0..639 and 768..2047). All 9 assertions passed.
