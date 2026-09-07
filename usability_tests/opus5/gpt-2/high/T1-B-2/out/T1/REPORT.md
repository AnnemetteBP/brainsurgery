# T1 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T1/plan.yaml` (output: `out/T1/model.safetensors`, 121 tensors)

- **Number of times you executed the script or plan:** 2 (both succeeded; the
  second run was only to capture the exit code and the tail of the log, the
  first run had already produced the correct output)

- **Which executions failed, and why (one line each):** none.

- **Pitfalls or surprises you hit (one line each):**
  - The renumbering collision hazard is real: `move` refuses an existing
    destination, so the order matters. Deleting blocks 2/5/8 first and then
    moving in *ascending* target order (3->2, 4->3, 6->4, 7->5, 9->6, 10->7,
    11->8) means each destination slot is always already vacated; the plan
    would have aborted with a destination-exists error otherwise.
  - Dots must be escaped in the `from`/`of` regexes (`h\.3\.(.*)`) but the `to`
    side is a replacement template, so it is written literally (`h.2.\1`).
  - `h\.(9|10|11)\..*` is safe only because matching is full-match; a partial
    match would have made `h\.1\..*` also catch `h.10.*` / `h.11.*`.
  - `attn.bias` is a causal-mask buffer, not a bias vector, but since the whole
    block is renamed by a single `(.*)` capture it needs no special handling.

- **Anything in the task text or documentation that was unclear:**
  - The README documents capture-group rewriting (`\1`, `\g<0>`) only under
    `assert: equal`, saying it works "exactly like `to` in copy/move"; the
    `help` text for `move` itself shows no capture-group example, so I had to
    infer that a pattern `move` with backreferences is supported. It is.
  - `output.path` ending in `.safetensors` gives a single file (no sharding);
    the README says this only indirectly, by saying directory-like paths
    shard.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~5 minutes, mostly reading
  `docpack/README.md` and the `move`/`delete`/`assert` entries of `help.txt`.
