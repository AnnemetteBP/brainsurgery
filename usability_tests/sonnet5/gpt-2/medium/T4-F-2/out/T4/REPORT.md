# Participant self-report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`.
- Number of times you executed the script or plan: 1.
- Which executions failed, and why (one line each): none — the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None specific to this task; the key precondition (identical non-MLP tensors across all three checkpoints) held as stated, so the verification step passed without needing to debug a mismatch.
- Anything in the task text or documentation that was unclear: no — the formula, tensor set, and required checks were unambiguous.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, used directly in a plain script rather than through mergekit's task-arithmetic YAML config. The formula only needs three checkpoint loads, a key-set/value-equality check, and a per-tensor float32 combination for 48 named tensors — writing that by hand made the "verify against the *unmodified* base" ordering and the three required checks (shared-tensor verification, exactly-48-merged, exactly-160-output) explicit and directly auditable in code, rather than trusting mergekit's YAML key-matching and per-model weighting to express the same order-sensitive formula correctly.
- Approximate time spent, if you can tell: a few minutes (single pass, no debugging needed).
