# T2 (OLMo-1B-0724-hf, condition B) participant self-report

- Final artifact path: `out/T2/plan.yaml` (output: `out/T2/model.safetensors`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source reference to resolve to exactly one tensor, so the plan needs one concat per tensor per layer (64 concat/delete/move triples), generated from a loop rather than a regex-with-capture-group rewrite.
  - Since `concat`'s destination must not exist yet, each tensor is rebuilt under a `tmp.` name, the original is deleted, then the temp is moved back to the original name.
  - Tensor names are full-match regexes, so dots were escaped in `from`/`target` references; `to` names are literal.
- Anything in the task text or documentation that was unclear:
  - `help.txt` shows `concat: { from: , to: a::xy, dim: 0 }` and `split: { ... to: , ... }` with the list arguments rendered empty, so the list syntax had to be inferred from the second example.
  - Whether `output.path` ending in `.safetensors` yields a single file (as opposed to a sharded directory) is only implied by the README; it did produce a single file.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 5 minutes (one plan run, 1-2 minutes of that was the run itself).
