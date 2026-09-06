# T4 report

- Final artifact path: `out/T4/solution.py` (invoked via `out/T4/run.sh`), output at `out/T4/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first run succeeded
- Pitfalls or surprises you hit (one line each): none; the 48 MLP tensor names
  and the non-MLP identity check were straightforward to enumerate from the
  task spec.
- Anything in the task text or documentation that was unclear: none.
- Tools used (condition F): `safetensors` 0.5.3 (load/save) and `torch` 2.14.0
  (float32 arithmetic, tensor comparison via `torch.equal`) — used directly
  rather than mergekit's task-arithmetic YAML recipe because this task's hard
  requirement is verifying the frozen-backbone precondition (identical
  non-MLP tensors across all three checkpoints) before merging, which a
  plain script checks explicitly and fails loudly on; mergekit's recipe
  format doesn't expose that check as a first-class step.
- Approximate time spent, if you can tell: a few minutes.
