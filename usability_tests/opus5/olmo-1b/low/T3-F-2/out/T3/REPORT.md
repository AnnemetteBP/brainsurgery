# T3 self-report

- Final artifact path: `out/T3/solution.py` (output in `out/T3/`, 10 shards + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each):
  - The obvious trap is an over-broad `.*weight` pattern; I used an anchored regex on
    `model.layers.<i>.{self_attn.{q,k,v,o}_proj,mlp.{gate,up,down}_proj}.weight` with escaped dots,
    and let the "exactly 112 bfloat16" assertion catch any drift.
  - `lm_head.weight` and `model.embed_tokens.weight` are 412 MB each, above the 256 MiB shard
    budget, so the packer needs an explicit oversized-tensor-alone case rather than plain greedy fill.
- Anything in the task text or documentation that was unclear:
  - The exact shard assignment is not specified beyond the size budget (order of packing, shard file
    naming). I packed greedily in the input index's key order and used the HF-standard
    `model-000NN-of-000MM.safetensors` naming.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3, plain script. The task is a
  per-tensor dtype policy plus a sharding policy; both are exactly and auditably expressible in
  ~20 lines of tensor code. `transformers.save_pretrained(dtype=...)` was the suggested route but it
  applies one dtype to the whole model, so it cannot express "bf16 projections, fp32 embeddings"
  without post-hoc per-tensor surgery anyway, and it re-derives the state dict through a model class.
- Approximate time spent, if you can tell: a few minutes.
