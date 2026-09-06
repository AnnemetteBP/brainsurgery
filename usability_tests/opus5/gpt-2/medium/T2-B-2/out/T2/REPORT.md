# T2 self-report (condition B: BrainSurgery plan)

- Final artifact path: `out/T2/plan.yaml` (output written to `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single run passed
  all in-plan asserts and wrote 160 tensors.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each `from` reference to resolve to exactly one tensor, so
    the plan cannot be written once with a layer capture; it is 12 unrolled
    layer blocks (3 concats + 3 deletes + 3 moves each).
  - `copy`/`concat` destinations must not already exist and `move` forbids
    slicing, so rebuilding in place needs the concat -> delete -> move dance
    through a scratch name (`pruned.<i>.*`).
  - Conv1D layout means c_attn heads are *column* blocks (dim 1) while c_proj
    heads are *row* blocks (dim 0); the two tensors prune on different axes.
  - `delete` takes a pattern, so dots must be escaped (`h\.0\.attn\.c_attn\.weight`)
    to avoid matching more than intended; `attn.c_proj.bias` and the `attn.bias`
    mask buffer must stay untouched.
- Anything in the task text or documentation that was unclear: the task text was
  explicit about the kept column/row ranges, which removed all ambiguity. The
  README does not state that `concat` sources must each resolve to a single
  tensor; that only appears in `help: concat`.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~5 minutes, mostly reading the doc pack.
