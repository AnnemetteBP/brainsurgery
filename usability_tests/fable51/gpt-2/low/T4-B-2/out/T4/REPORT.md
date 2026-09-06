# T4 self-report (condition B, GPT-2 124M)

- Final artifact path: `out/T4/plan.yaml` (output checkpoint `out/T4/model.safetensors`, 160 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `failed_assertion` — my `writes: { ..., is: 0 }` check for untouched tensors fired (`base::h.0.attn.bias had writes=1`); the initial load counts as one write. Everything before that assert (shared-tensor verification, task vectors, merge, 48/160 counts) had passed. Fixed by asserting `writes is 1` for untouched tensors and `writes is 3` (load + two `add_`) for the 48 merged ones.
  - Execution 2: success.
- Pitfalls or surprises you hit (one line each):
  - Write counters start at 1 after loading, so "untouched" means `writes: is 1`, not 0.
  - With three inputs, every destination must carry the `base::` alias (task-vector scratch tensors were created under `base::tv1.*` / `base::tv2.*` and deleted afterwards) so the output alias can be inferred.
  - To check that MLP tensors exist with matching shape/dtype in the fine-tunes without asserting equal values, I used `equal` with a huge `eps`; a dedicated "same shape/dtype" cross-alias check would be cleaner.
  - Task-vector ordering was handled by materializing both `ft - base` differences before the first `add_` into the base.
- Anything in the task text or documentation that was unclear:
  - The help text for `writes`/`reads` does not say that loading counts as a write.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: ~5 minutes
