# T5 self-report (condition B: BrainSurgery plan)

- **Final artifact path:** `out/T5/plan.yaml` (output checkpoint in `out/T5/`,
  5 shards + `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single
  execution passed all asserts and wrote the sharded output.
- **Pitfalls or surprises you hit (one line each):**
  - `fan_in_fan_out = true` / Conv1D `[in, out]`: `B @ A` is `[2304, 768]` and
    must be transposed (`permute: order: [1, 0]`) before `add_` into the
    `[768, 2304]` base weight.
  - Output alias inference: every write has to land on one alias, so the
    `matmul` destinations were named `base::loramerge...` rather than left on
    the `lora` alias.
  - Intermediates are ordinary tensors in the model, so an explicit
    `delete` of `base::loramerge\..*` is needed before the output is written.
  - Regex refs are full-match, so dots must be escaped in `of:`/`from:`
    patterns; the rewrite side (`from_b`, `to`) is a replacement string, not a
    pattern, and uses `\1` with plain dots.
  - `100MB` in the plan is binary (104,857,600 bytes), which is exactly the
    task's budget; `wte.weight` (154 MB) landed alone in its own shard as
    documented.
- **Anything in the task text or documentation that was unclear:** nothing
  blocking. The README's note that a ternary transform resolves `from_b` and
  `to` as rewrites of the `from_a` captures is only stated in the interfaces
  reference, not next to `matmul`'s own help.
- **Tools used (condition F):** n/a
- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading
  `README.md` and `help.txt`.
