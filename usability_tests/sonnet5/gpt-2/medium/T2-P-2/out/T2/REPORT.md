## Participant self-report

- Final artifact path: `out/T2/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - Had to keep in mind GPT-2's Conv1D `[in, out]` layout (transpose of `nn.Linear`), so `c_attn` heads live on axis 1 (columns) while `c_proj` heads live on axis 0 (rows) of the input side.
  - `c_attn.weight`/`c_attn.bias` need the head removed independently from each of the three 768-wide q/k/v segments, not once from the full 2304-wide axis.
- Anything in the task text or documentation that was unclear: no, the required column/row ranges in TASK.md matched exactly what dropping head 5's 64-wide block from each q/k/v segment (and from `c_proj`'s row blocks) produces, which let me self-check the derived indices against the spec before writing the file.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: a few minutes.
