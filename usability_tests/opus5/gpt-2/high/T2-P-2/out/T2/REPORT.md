# Participant self-report — T2 (Condition P)

- **Final artifact path:** `out/T2/solution.py` (output: `out/T2/model.safetensors`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - Conv1D `[in, out]` layout means the head axis differs per tensor: columns (dim 1) for `c_attn.weight`, rows (dim 0) for `c_proj.weight` — slicing both on the same axis would have been the easy mistake.
  - `c_attn` is fused `[q | k | v]`, so head 5 has to be dropped three times at offsets 0, 768 and 1536, not once.
  - `h.<i>.mlp.c_proj.weight` (`[3072, 768]`) shares the `c_proj` suffix with the attention output projection; I matched exact tensor names rather than a suffix pattern so it could not be caught by accident.
  - `h.<i>.attn.bias` is the `[1, 1, 1024, 1024]` causal-mask buffer, not a per-head bias — left untouched.
  - `index_select` results needed `.contiguous()` before `save_file`; I also carried the input's `{"format": "pt"}` metadata across so the output header matches the input's.
- **Anything in the task text or documentation that was unclear:** nothing. The task spelled out the exact column/row ranges to keep, which I used as a cross-check against the head arithmetic (`head_dim = 64`, prune head 5) rather than hardcoding — the script asserts the two agree before touching any tensor.
- **Tools used (condition F):** n/a — condition P: Python 3.13, `torch==2.14.0`, `safetensors==0.5.3` only.
- **Approximate time spent, if you can tell:** ~5 minutes: read the safetensors JSON header directly (via `od`/`dd`) to confirm the flat `h.<i>.` key names, dtypes and the 160-tensor count before writing a line of the script, then one write and one run.
