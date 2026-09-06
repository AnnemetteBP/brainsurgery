## Participant self-report

- Final artifact path: `out/T5/plan.yaml`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single execution
  against `out/T5/plan.yaml` succeeded.
- Pitfalls or surprises you hit (one line each):
  - `assert: { count: { of: ..., is: 0 } }` raises an error itself when the
    reference matches zero tensors (`count.of matched zero tensors`), so it
    cannot be used to assert "no tensor matches"; used
    `assert: { not: { exists: ... } }` instead (caught and fixed before the
    single logged execution, via a throwaway scratch plan outside `out/`).
  - Ternary transforms (`matmul`) and the plain mapping transforms
    (`permute`, `scale`, `add_`) all support the same regex-capture rewrite
    (`\1`) across `from`/`from_a`/`from_b`/`to`, which let one transform block
    per step cover all 12 layers instead of writing 12 blocks by hand;
    confirmed this against a tiny synthetic checkpoint before relying on it.
  - Intermediate tensors (`_tmp.<i>.delta_lin`, `_tmp.<i>.delta_conv1d`,
    `_tmp.<i>.delta_scaled`) had to be created under the `base` alias, not a
    third alias, and deleted (not just left over) before `output`, otherwise
    output-alias inference would see writes to more than one alias.
- Anything in the task text or documentation that was unclear: no — the
  Conv1D `[in, out]` vs `nn.Linear` `[out, in]` layout and the resulting need
  to transpose `B @ A` was stated plainly in TASK.md and matched what
  `fan_in_fan_out = true` means in `adapter_config.json`.
- Tools used (condition F): n/a (condition B).
- Approximate time spent, if you can tell: ~15 minutes, most of it building
  and testing the plan against small synthetic checkpoints in `/tmp` before
  running it against the real inputs.
