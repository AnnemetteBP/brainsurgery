# T3 self-report (condition B — BrainSurgery plan)

- **Final artifact path:** `out/T3/plan.yaml` (output checkpoint in `out/T3/`:
  9 shards + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none — the plan passed on
  its first execution.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious trap is pattern breadth: `.*weight` would have swept up
    `gpt_neox.embed_in.weight`, `embed_out.weight` and every layer-norm weight,
    so I spelled the four projection module names out explicitly
    (`attention.(query_key_value|dense)`, `mlp.(dense_h_to_4h|dense_4h_to_h)`)
    rather than matching on the `.weight` suffix.
  - The buffer names collide with real parameters: `…attention.bias` is a
    causal-mask buffer while `…attention.dense.bias` and
    `…attention.query_key_value.bias` are parameters that must survive. A delete
    pattern must be anchored so `bias` is the final segment right after
    `attention.`; I used `attention\.(?:bias|masked_bias|rotary_emb\.inv_freq)`
    under full-match semantics and asserted afterwards that the 32 projection
    biases were still there.
  - Ordering matters for the cast pass. I cast `.*` to float32 first and only
    then cast the 64 projections down to bfloat16; doing it the other way round
    would have upcast the bfloat16 matrices straight back to float32.
  - Casting had to be done with `cast_` (in place), not `cast`. `cast` writes
    *new* destination tensors, which would have changed the names and the
    state-dict ordering that the shard packing depends on.
  - The buffers had to be deleted *before* the blanket `cast_: {target: '.*'}`,
    since `gpt_neox.layers.<i>.attention.bias` is uint8 and casting it to
    float32 would have been meaningless work on 16 × 4 MiB mask tensors.
  - "at most 256 MiB" maps cleanly onto the tool's binary shard units, so
    `shard: 256MB` is literally 268,435,456 bytes; the two 412 MB embedding
    matrices exceed that on their own and the writer put each one alone in its
    own shard, which is what the task asks for.
  - There is no assert operator that counts tensors *by dtype*, so "exactly 64
    tensors are bfloat16" had to be expressed as a pair of complementary
    checks: the 64 tensors matching the projection pattern are bfloat16, and
    every tensor matching the negative-lookahead complement (132 of them) is
    float32. Together those pin the dtype of all 196 tensors.

- **Anything in the task text or documentation that was unclear:**
  - The README documents `count`, `dtype` and `exists` but not whether `dtype`
    applies to *every* match of a multi-match reference or just the first. I
    relied on it being every match (which the run's behaviour is consistent
    with) and additionally pinned the match counts with `count`, so the pair of
    assertions is sound either way.
  - The task states the shard budget in bytes and the tool states it in binary
    units; that they agree exactly (`256MB` = 256 × 1024 × 1024) is convenient
    but worth stating, since `MB` is decimal in most other tools.
  - Not stated, but it turned out not to matter: whether shard packing follows
    the input checkpoint's key order. It does, and the header order of the input
    is alphabetical.

- **Tools used (condition F):** n/a — condition B. Only `brainsurgery` itself,
  plus `od`/`dd`/`grep` to read the safetensors headers of the input and of the
  produced shards for verification.

- **Approximate time spent, if you can tell:** roughly 10 minutes, most of it
  reading the transform and assert help output and enumerating the input's key
  set before writing the plan. The plan run itself took ~15 s.
