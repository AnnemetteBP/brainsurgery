# T2 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T2/plan.yaml` (plan), `out/T2/model.safetensors` (output)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):**
  - None. The single execution passed every `assert` and wrote the output.

- **Pitfalls or surprises you hit (one line each):**
  - `concat` requires each `from` reference to resolve to exactly one tensor, so it cannot be
    driven by a regex over all 12 layers; the plan has to enumerate every layer explicitly
    (36 concat/delete/move triples, generated so all layers are byte-for-byte uniform).
  - Names must not change, and `move`/`concat` destinations must not already exist, so the
    rebuild has to go through a temporary name: `concat -> pruned.<name>`, `delete <name>`,
    `move pruned.<name> -> <name>`.
  - Source references are full-match regexes, so `h.0.attn.c_attn.weight` has wildcard dots;
    I escaped them (`h\.0\.attn\.c_attn\.weight`) everywhere a reference is *matched*, but left
    them unescaped in `to:` destinations, which are rewrite templates and would otherwise carry
    the backslashes into the tensor name.
  - Conv1D layout is the real trap: `c_attn.weight` is `[in, out]`, so heads are *column* blocks
    (dim 1) and head 5 is columns 320:384 within each of the three 768-wide q/k/v segments
    (i.e. 320:384, 1088:1152, 1856:1920 of the fused tensor), while `c_proj.weight` is the
    transpose situation and heads are *row* blocks (dim 0). Slicing the wrong axis on either one
    still yields a loadable checkpoint with the right shape but garbage attention.
  - `attn.bias` is the causal mask buffer, not a head-bearing bias; only `c_attn.bias` is pruned.
    An unanchored pattern like `.*attn.*bias` would wrongly hit `attn.bias` and `c_proj.bias`.

- **Anything in the task text or documentation that was unclear:**
  - The docs do not say whether an `assert: shape` whose `of` matches several tensors checks all
    matches or errors out. Rather than risk it, I wrote one single-tensor shape assert per layer
    (36 of them) instead of three pattern-based ones.
  - Likewise `count: { is: 0 }` on a non-matching pattern is not documented, so the
    "no temporaries left behind" check is written as `not: { exists: ... }`.
  - The task text was unambiguous about the column ranges to keep, which removed the main
    source of doubt about the fused q/k/v layout.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading `help.txt` for the
  exact `concat`/`move` contracts and verifying the result.

## Verification performed after the run

Beyond the in-plan asserts, I checked the written file directly against the input by comparing
raw safetensors byte ranges (no Python, per the condition rules):

- Output has exactly 160 tensors, key set identical to the input, all dtypes `F32`.
- All 12 layers: `c_attn.weight` `[768, 2112]`, `c_attn.bias` `[2112]`, `c_proj.weight` `[704, 768]`.
- All 124 untouched tensors are bit-identical to the input.
- 312 block-level byte comparisons across all 12 layers confirmed the six surviving
  `c_attn` column blocks and the two surviving `c_proj` row blocks land in the right order,
  and that head 5's slice is absent from the output.
