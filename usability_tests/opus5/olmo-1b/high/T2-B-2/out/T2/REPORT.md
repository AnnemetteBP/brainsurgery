# T2 self-report (condition B, BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (plan), `out/T2/model.safetensors` (output, 114 tensors)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** None; the single
  execution succeeded and every assert in the plan passed.

- **Pitfalls or surprises you hit (one line each):**
  - `concat` needs each source to resolve to exactly one tensor, so the obvious
    "split then concat" route would have meant 64 hand-written blocks; instead I
    used `copy` (a correctly *shaped* slice) + two sliced `assign`s, which are
    regex-driven and cover all 16 layers in one transform each.
  - Destinations of `copy`/`move` must not exist and cannot be sliced, so the
    pruned tensors have to be built under a scratch name (`<proj>.pruned`) and
    only then `delete`d + `move`d onto the original names.
  - `assert: count: is: 0` felt risky as a "no leftovers" check (a reference that
    matches nothing may error rather than count zero), so I used
    `not: { exists: ... }` instead.
  - Escaping dots in the `from` regex matters for not overmatching, but the `to`
    side is a name template, so `\1`/`\2` capture references there are literal
    text with plain dots.
  - The output path needs the `.safetensors` suffix to get a single file; a
    suffix-less path would have been treated as a directory and sharded.

- **Anything in the task text or documentation that was unclear:** Nothing
  blocking. The task text is explicit about the layout (`[out, in]`, heads as
  row blocks for q/k/v and column blocks for o), which is exactly the part that
  is normally guesswork. The README's `subtract_` example `from: 'a::[:, :10]'`
  is ambiguous about whether an alias-less `expr::[slice]` form is allowed; I
  sidestepped it by always writing the explicit `model::name::[slice]` form.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `docpack/help.txt` for `split`/`concat`/`copy`/`assign` slicing and
  destination rules; the plan run itself took 12 s.

## What the plan does

Head 5 of 16 (dims 640..767) is dropped in every layer. For each of the 16
layers, driven by regex captures so each transform covers all layers at once:

1. `copy` a `[1920, 2048]`-shaped (resp. `[2048, 1920]`) slice of the original
   into `<proj>.pruned` to get a destination of the right shape;
2. `assign` the head 0-4 block (`[:640]` / `[:, :640]`) into its front;
3. `assign` the head 6-15 block (`[768:]` / `[:, 768:]`) directly behind it;
4. `delete` the originals and `move` the scratch tensors onto their names.

Required checks are `assert` transforms placed after the edit and before
`output`: the four layer-0 shapes and `count: 114`, plus preflight shape/count
asserts on the input layout and extra checks on layer 15, dtype, leftover
scratch names and the untouched MLP tensors.

## Verification performed

Independently of the plan's own asserts: the output has exactly 114 tensors with
a key set identical to the input's, and every tensor is bit-exact against the
expected result computed from the input shards (q/k/v rows and o_proj columns
gathered at `0..639, 768..2047`; all other tensors unchanged) - 0 mismatches
over all 114 tensors, dtypes all F32.
