# T4 participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`.
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None functionally; the only care point was reading tensors with `safe_open(..., framework="pt")` and doing the `base + lambda*(ft1-base) + lambda*(ft2-base)` arithmetic in float32 before casting back to float16, per the spec, and computing every task vector against the original `base` dict rather than against a mutated one (built the merged dict as a fresh `out` mapping to avoid the ordering hazard by construction).
- Anything in the task text or documentation that was unclear: no, the spec (verify first, formula, output shape) was precise enough to implement directly.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, used directly in a plain script rather than through `mergekit`'s task-arithmetic YAML. The required checks are exact per-tensor assertions (same key set; bit-identical non-MLP tensors across all three checkpoints; exactly 64 merged; exactly 244 output tensors), which are simplest to express and to fail loudly on as explicit Python assertions than to configure through a merge-toolkit YAML; `torch`/`safetensors` give full control over which tensors are touched and the float32-then-float16 cast order.
- Approximate time spent, if you can tell: a few minutes (single-pass implementation, verified once against the base/ft1/ft2 inputs before finishing).
