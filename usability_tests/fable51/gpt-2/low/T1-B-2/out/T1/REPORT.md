# Participant self-report: T1 (GPT-2, condition B)

- Final artifact path: `out/T1/plan.yaml` (output checkpoint `out/T1/model.safetensors`, 121 tensors)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  - Execution 1: `failed_assertion` / `no_match`. The check "no tensor of blocks 9, 10, 11 remains" was written as `assert count of 'h\.(9|10|11)\..*' is 0`; `count` raises `matched zero tensors` before comparing, so a zero-count assertion can never pass. The edit itself (delete + 7 moves) had already succeeded; no output was written.
- Pitfalls or surprises you hit (one line each):
  - `assert count ... is: 0` is unusable for absence checks; I rewrote it as a negative lookahead: `count of '(?!h\.(9|10|11)\.).*' is 121` together with `count of '.*' is 121`.
  - Renumbering by ascending source index (3->2, 4->3, 6->4, ...) guarantees every destination is free because each destination index is below its source and the old slot was either deleted or already moved.
  - `move` with regex `from` and `\1` backreference in `to` renames a whole block (13 tensors) in one transform.
- Anything in the task text or documentation that was unclear:
  - README/help do not say that a reference matching zero tensors is an error even inside `assert count`; a note or a `is: 0` special case would avoid the retry.
- Tools used (condition F): n/a
- Approximate time spent, if you can tell: about 3 minutes
