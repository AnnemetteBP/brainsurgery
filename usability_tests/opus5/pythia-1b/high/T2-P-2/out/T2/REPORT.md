# T2 self-report (condition P)

- **Final artifact path:** `out/T2/model.safetensors` (script: `out/T2/solution.py`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single execution
  passed every check and wrote the output.
- **Pitfalls or surprises you hit (one line each):**
  - The fused `query_key_value` layout is the real trap: GPT-NeoX interleaves
    q/k/v *inside* each head's 768-row block, so head 5 is one contiguous
    3840..4607 row span; treating the tensor as `[q | k | v]` segments would
    need three separate cuts and give a different (wrong) result of the same shape.
  - The two head-bearing axes differ: rows for the input-side fused projection,
    columns for `attention.dense.weight`, which consumes heads as 256-wide
    column blocks. The bias is not per head and stays `[2048]`.
  - `attention.dense.weight` is square `[2048, 2048]`, so a wrong-axis slice
    still produces a loadable checkpoint with a plausible shape; I asserted the
    exact post-slice shape `[2048, 1792]` rather than just "one dim shrank".
  - Name matching had to be anchored: `attention.dense.weight` must not catch
    the MLP tensors, so I used fully anchored regexes and then asserted the
    match counts (32 qkv tensors, 16 dense weights over 16 layers) so a pattern
    that under- or over-matched would fail loudly instead of silently passing.
  - Fancy indexing with an index tensor returns a fresh tensor, so nothing was
    a view into the input; I still called `.contiguous()` before saving because
    safetensors rejects non-contiguous or memory-sharing tensors.
  - Preserved the input's safetensors metadata header and asserted no dtype
    changed, since grading is bit-exact on values and dtypes.
- **Anything in the task text or documentation that was unclear:** nothing. The
  task spelled out the interleaved layout and the exact index ranges to keep,
  which removed the only genuinely ambiguous part of the job.
- **Tools used (condition F):** n/a (condition P: torch 2.14.0 + safetensors 0.5.3).
- **Approximate time spent, if you can tell:** a few minutes; one pass to read
  the task and inspect `config.json`, one to write the script, one execution.
