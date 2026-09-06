# T3 self-report (condition B, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T3/plan.yaml` (output checkpoint in `out/T3/`:
  10 shards + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none; the single
  execution passed all asserts and wrote the output.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious hazard is regex overreach: `.*weight` would also hit
    `model.embed_tokens.weight` and `lm_head.weight`, so I anchored on
    `model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight`
    and checked the complement with a negative-lookahead `dtype` assert rather
    than trusting the positive pattern alone.
  - Shard budget units: `256MB` in brainsurgery is binary (256 x 1024^2 =
    268,435,456 bytes), which is exactly the task's budget, so `shard: 256MB`
    is right and no MB/MiB conversion was needed.
  - Two full layers of projections come to exactly 268,435,456 bytes in
    bfloat16, so the shards land exactly *on* the budget rather than under it;
    that is fine (the rule is "at most"), but it made me double-check the
    packing arithmetic instead of eyeballing file sizes.
  - `cast_` (in-place) rather than `cast` (creates new destinations) was the
    right choice: names must not change, and in-place targets also let the
    engine infer the output alias unambiguously.
  - The task mentions dropping buffers, but this checkpoint has none, so the
    plan deletes nothing — the `count: 114` asserts before and after guard
    against an accidental deletion.

- **Anything in the task text or documentation that was unclear:**
  - The objective paragraph is written generically ("drop non-parameter
    buffers", "norms, biases") while the Input section says this checkpoint has
    no buffers, no norms and no biases. The concrete sections win, but the
    mismatch invites writing a `delete` transform that would match nothing.
  - The docs say a `dtype` assert "succeeds if the tensor has the given dtype"
    without stating what happens for a multi-match reference; I relied on it
    meaning "every match", which the run's behaviour is consistent with, but
    it would be worth spelling out.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** roughly 5 minutes — read the
  README/help for `cast_`, `assert.dtype/count` and the sharding semantics,
  listed the input keys to confirm 2 + 16x7 = 114, wrote the plan, ran it once,
  then verified the result independently.

## Verification performed after the run

Independent of the plan's own asserts, I re-read `out/T3` and confirmed:
114 tensors in the index and on disk, 112 BF16 / 2 F32, every shard's tensor
data <= 268,435,456 bytes with each 412 MB embedding matrix alone in its own
shard, key set identical to the input, and every tensor bit-exact against
`input.to(torch.bfloat16)` for the projections and against the untouched input
for the rest.
