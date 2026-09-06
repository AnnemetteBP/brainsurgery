# Participant self-report

- Final artifact path: `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded
- Pitfalls or surprises you hit (one line each):
  - `attention.masked_bias` is stored as float16 in the input, not some obviously-scalar
    dtype, so it can't be told apart from a real parameter by dtype alone; it has to be
    matched by name.
  - Embeddings/output projection are 206 MB in the fp16 input but ~393 MiB once upcast to
    float32 for the output, which is what actually pushes them over the 256 MiB shard
    limit and forces them into their own shards — the "already oversized" framing in the
    task text is about the *output* dtype, not the input.
  - A naive `.*weight` or `.*bias` regex would also catch `embed_in.weight`,
    `embed_out.weight`, layer-norm weights/biases and the projection biases, all of which
    must stay float32; matched the 4 projection-weight submodule paths explicitly instead
    of relying on a generic "weight" suffix.
- Anything in the task text or documentation that was unclear: no, the tensor list, shapes
  and per-tensor dtype/shard rules in TASK.md were sufficient to write the regexes and
  shard packer directly.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain
  script (`load_file`/`save_file`). Considered `transformers`' sharded `save_pretrained`
  with a dtype argument, but it applies one dtype to the whole model rather than a
  per-tensor split between bfloat16 projections and float32 everything else, so it can't
  express requirement 1+2 in one pass without a manual state-dict rewrite first — at that
  point a direct script is simpler and keeps the shard-packing logic (oversized-tensor
  handling) fully explicit and inspectable. Did not use `mergekit` or `peft`: this task is
  a single-checkpoint dtype/shape rewrite, not a merge or adapter operation, so neither
  toolkit's abstractions (merge configs, LoRA loading) fit.
- Approximate time spent, if you can tell: one solve-and-verify pass, no iteration needed.
