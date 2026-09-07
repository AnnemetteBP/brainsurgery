# T2 self-report (condition P)

- Final artifact path: `out/T2/solution.py` (output `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution passed all checks.
- Pitfalls or surprises you hit (one line each):
  - GPT-2's Conv1D `[in, out]` layout means head blocks are *columns* of `c_attn` but *rows* of `c_proj`; using the `nn.Linear` intuition would transpose both.
  - `c_attn` is the fused `[q | k | v]` projection, so head 5 has to be dropped once per 768-wide segment (three disjoint column ranges), not once overall.
  - Advanced indexing returns non-contiguous views, so I called `.contiguous()` before `save_file` to avoid a safetensors write error.
- Anything in the task text or documentation that was unclear: nothing; the explicit index ranges made the intended layout unambiguous and served as a cross-check on the computed keep-indices.
- Tools used (condition F): n/a (condition P: torch + safetensors only)
- Approximate time spent, if you can tell: a few minutes; one read of the task, one script, one run.
