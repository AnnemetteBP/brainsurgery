# T1 self-report (condition B, GPT-2 124M)

- Final artifact path: `out/T1/model.safetensors` (plan: `out/T1/plan.yaml`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first execution succeeded (exit 0, 121 tensors written).
- Pitfalls or surprises you hit (one line each):
  - Renumbering collision hazard handled by moving in ascending source order (3→2, 4→3, 6→4, 7→5, 9→6, 10→7, 11→8) after deleting 2/5/8, so every destination index is vacant when written; `move` refuses existing destinations, so a wrong order would fail rather than clobber.
  - Regex tensor refs are full-match, so dots in `h.<i>.` must be escaped (`h\.3\.(.*)`) and `\1` is used in `to` to keep the rest of the name.
  - `count` with `of: '.*'` is the simplest way to assert the total tensor count.
  - Verified separately with a read-only plan (`out/T1/verify.yaml`, no `output` section) using `equal` with capture rewrites: non-block tensors and blocks 0/1 identical to the input, and each new block `k` bit-identical to its old block; that plan was run once and writes nothing.
- Anything in the task text or documentation that was unclear:
  - Nothing blocking. `help.txt` says the assert `count` payload is `of`/`is`, but does not state whether `of` counts matches across all aliases or only the default alias; with one input it did not matter.
- Tools used (condition F): n/a (condition B)
- Approximate time spent, if you can tell: about 3 minutes (docs reading ~2 min, plan + one run + verification ~1 min).
