## Participant self-report

- Final artifact path: `out/T2/solution.py` (invoked via `out/T2/run.sh`), output `out/T2/model.safetensors`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first run succeeded
- Pitfalls or surprises you hit (one line each): none — the task spec gave the exact row/column ranges to keep (0..3839/4608..6143 for qkv, 0..1279/1536..2047 for dense columns), so it was a direct translation into slice-and-cat operations, no head-index arithmetic needed on my end
- Anything in the task text or documentation that was unclear: no
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 for direct load/slice/save of the state dict. Chose a plain script over `transformers.PreTrainedModel.prune_heads` because that API prunes by internal head index against its own q/k/v-splitting logic, which is an extra layer of indirection versus the byte-exact row/column ranges given in the spec; a direct slice on the raw tensors is easier to verify against the spec and to assert shapes on before writing.
- Approximate time spent, if you can tell: a few minutes
