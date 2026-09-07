# T4 report

- **Final artifact path:** `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`.
- **Number of times you executed the script or plan:** 1.
- **Which executions failed, and why:** none; the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - None specific to this task; the main care point was doing the equality
    check and the arithmetic in the right order (task vectors against the
    unmodified `base`, not against each other or a running merge), which the
    spec calls out explicitly.
- **Anything in the task text or documentation that was unclear:** no.
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — load/save the three input checkpoints and the
    output checkpoint directly, no intermediate format needed.
  - `torch` 2.14.0 — float32 accumulation for the task-arithmetic sum and
    cast back to float16; `torch.equal`/shape/dtype comparisons for the
    precondition check.
  - Did not use `mergekit`'s task-arithmetic YAML config: it would compute
    the same result but hide the per-tensor MLP selection and the mandatory
    "180 tensors must be bit-identical across all three checkpoints" abort
    check behind library internals, making the two required checks (shared-
    tensor verification, exact merged-tensor count) harder to state and
    verify directly. A ~110-line script keeps both checks explicit and
    inspectable.
- **Approximate time spent, if you can tell:** a few minutes (single-pass
  script, ran clean on first execution).
