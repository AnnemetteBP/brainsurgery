# Participant self-report — T5 (Condition B, Pythia-1B)

- **Final artifact path:** `out/T5/plan.yaml`

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):**
  - None; the single execution succeeded on the first attempt.

- **Pitfalls or surprises you hit (one line each):**
  - `add_`/`add` require exact shape/dtype/device match on both operands, so
    the fp32 `matmul` result (`lora_B @ lora_A`) has to be `scale_`'d and then
    explicitly `cast` to float16 as a separate new tensor before it can be
    `add_`'d into the float16 base weight in place.
  - `matmul`/`add_`/`cast`/`scale_` all support the same regex-capture
    rewrite model as `copy` (`from_a`/`from_b`/`to` share `\1`), so all 16
    layers can be merged with one `matmul` + one `scale_` + one `cast` + one
    `add_`, instead of 16 repeated blocks.
  - Intermediate tensors (`delta.<i>.weight`, `delta_fp16.<i>.weight`) have to
    be written into the same alias (`base`) as the destination weights so the
    output-alias inference stays unambiguous, and then explicitly `delete`d
    before saving so they don't end up in the output.
  - Directory-style `output.path` with `shard: 512MB` was sufficient to get
    sharded safetensors output with an index file; no manual shard bookkeeping
    was needed.

- **Anything in the task text or documentation that was unclear:**
  - TASK.md states the two embedding tensors are "206 MB each" and calls them
    out as exceeding the 512 MiB shard budget and therefore being "stored
    alone in its own shard," but 206 MB is well under 512 MiB — this reads as
    inconsistent/misleading, though it didn't affect the plan since sharding
    is handled automatically by `output.shard`.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~15 minutes (reading
  `docpack/README.md` and `docpack/help.txt` for `matmul`/`add_`/`cast`/
  `scale_`/assert semantics, writing the plan, and one verification run).
