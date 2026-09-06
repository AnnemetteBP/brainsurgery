# T2 self-report (Condition B: BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (output: `out/T2/model.safetensors`)

- **Number of times you executed the script or plan:** 2 executions of
  `out/T2/plan.yaml`. (Plus 3 executions of separate scratch plans that never
  wrote output: one to inspect layer-0 shapes and the tensor count before
  writing the plan, two to verify the result afterwards. Those scratch plans
  have been deleted.)

- **Which executions failed, and why (one line each):**
  - Execution 1 — `failed_assertion` / `no_match`: my own leftover-temporaries
    check `count: { of: 'pruned_tmp\..*', is: 0 }` aborted with
    `count.of matched zero tensors`; `count` treats zero matches as an error, so
    it cannot express "expected none". The surgery itself was correct up to that
    point; no output was written. Replaced with `not: { exists: ... }`.
  - Execution 2 — passed, wrote `out/T2/model.safetensors`.

- **Pitfalls or surprises you hit (one line each):**
  - `count: { is: 0 }` is not usable as a "nothing matched" assertion; the
    resolver raises before the count is compared. `not: { exists: ... }` is the
    right form.
  - `concat` requires each `from` reference to resolve to exactly one tensor, so
    it cannot be written once with a `\d+` layer pattern — the plan needs one
    concat per layer per tensor (48 of them, generated mechanically).
  - Names cannot change, but `concat`/`copy` destinations must not already
    exist, so the edit needs a three-step swap: concat into `pruned_tmp.<i>.*`,
    `delete` the originals by pattern, then `move` the temporaries back onto the
    original names using a capture group (`\1`) in the `to` rewrite.
  - The two head-bearing axes differ: rows (dim 0) for the fused
    `query_key_value` weight and bias, columns (dim 1) for `attention.dense.weight`,
    because `nn.Linear` stores `[out, in]` and the output projection consumes
    the head dimension on its input side.
  - The GPT-NeoX interleaved qkv layout means head 5 is a single contiguous
    768-row block (3840..4607), not three separate 256-row slices in `[q|k|v]`
    segments — the interleaving actually made the slicing simpler, but assuming
    `[q|k|v]` segments would have silently produced a loadable, wrong checkpoint.
  - I escaped every `.` in the match patterns; unescaped dots would still
    full-match here, but the `delete` patterns are broad enough (`\d+`) that I
    did not want to rely on that.
  - `attention.bias` (the `[1, 1, 2048, 2048]` causal mask buffer) is named like
    a projection bias; the `delete` patterns were anchored on
    `attention\.query_key_value\.` and `attention\.dense\.weight` so it was never
    at risk, and I checked the untouched tensors bit-for-bit afterwards.

- **Anything in the task text or documentation that was unclear:** Nothing that
  blocked me. The task text was unusually explicit about the layout (it gave the
  exact row and column ranges), which removed the main risk. Two documentation
  gaps: the README lists `count` without saying that zero matches is an error
  rather than a count of zero, and neither the README nor `help.txt` states
  whether `concat`/`split` support multi-match patterns — `concat`'s help does
  say each source must resolve to exactly one tensor, which I only noticed on a
  second read.

- **Tools used (condition F):** n/a — condition B, plan only.

- **Approximate time spent, if you can tell:** roughly 5 minutes of wall clock;
  most of it reading `help.txt` and generating the 71-transform plan, one
  minute of actual plan execution.

## What the plan does

For every layer `i` in 0..15:

1. `concat` rows `[0:3840]` and `[4608:6144]` of
   `attention.query_key_value.weight` along dim 0 into `pruned_tmp.<i>.qkv.weight`
   → `[5376, 2048]`.
2. The same on `attention.query_key_value.bias` → `[5376]`.
3. `concat` columns `[:, 0:1280]` and `[:, 1536:2048]` of
   `attention.dense.weight` along dim 1 into `pruned_tmp.<i>.dense.weight`
   → `[2048, 1792]`.

Then the 48 originals are deleted by pattern and the 48 temporaries are moved
back onto the original names.

Checks in the plan: input preconditions (244 tensors, 16 qkv weights, the three
layer-0 input shapes), then the required post-conditions — layer-0 shapes
`[5376, 2048]`, `[5376]`, `[2048, 1792]`, exactly 244 tensors, no leftover
temporaries — plus the same three shape checks for layers 1..15, the shapes of
two untouched tensors, and a `float16` dtype check.

## Verification performed after the run

With a separate read-only plan loading both the base and the output as two
aliases:

- all 196 tensors that are not head-bearing are bit-identical to the base
  (`assert equal` with a negative-lookahead pattern; the count of that pattern
  is 196, and 196 + 48 = 244);
- for all 16 layers, output qkv rows `[0:3840]` equal base rows `[0:3840]` and
  output rows `[3840:5376]` equal base rows `[4608:6144]` (weight and bias);
- for all 16 layers, output dense columns `[:, 0:1280]` equal base
  `[:, 0:1280]` and output `[:, 1280:1792]` equal base `[:, 1536:2048]`;
- shapes confirmed on layers 0, 7 and 15; dtype `float16` confirmed on the
  rebuilt tensors.
