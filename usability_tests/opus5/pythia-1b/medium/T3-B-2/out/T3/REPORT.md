# T3 self-report (condition B, Pythia-1B)

- **Final artifact path:** `out/T3/plan.yaml` (output checkpoint in `out/T3/`,
  9 shards + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none; the single run passed.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious hazard is regex overreach: `.*weight` would hit `gpt_neox.embed_in.weight`,
    `embed_out.weight` and every layer-norm weight, so the bf16 target is an explicit
    alternation over the four projection names anchored with escaped dots.
  - "Exactly 64 tensors are bfloat16" is not expressible as one assert, because `count`
    counts name matches and `dtype` checks a matched set; I encoded it as
    count(proj)==64 AND dtype(proj)==bf16 AND dtype(complement, via a negative-lookahead
    regex)==float32, which together pin the bf16 set to exactly those 64.
  - Ordering matters: delete the 48 buffers first, then `cast_ target: '.*' to: float32`,
    then downcast the projections. Casting `.*` before deleting would have tried to turn the
    uint8 causal-mask buffer into float32.
  - Shard budget: `256MB` in this tool is binary (256*1024*1024 = 268,435,456), which is
    exactly the task's limit, and the two 206 MB fp32 embedding matrices exceed it, so the
    writer put each alone in its own shard as documented.

- **Anything in the task text or documentation that was unclear:**
  - Nothing blocking. The README's note that shard budgets count tensor data only (not
    headers) and that an oversized tensor gets its own shard was exactly the detail needed;
    without it the 256 MiB rule versus the 412 MB shard files on disk would have looked wrong.
  - `assert: dtype` is documented in the singular ("the tensor"), but it accepts a pattern and
    checks every match, which is what makes the complement check possible; that could be
    stated explicitly.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes, most of it reading the doc pack
  and inspecting the input key/dtype list before writing the plan.
