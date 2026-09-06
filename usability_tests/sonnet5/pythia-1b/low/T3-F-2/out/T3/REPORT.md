## Participant self-report

- Final artifact path: `out/T3/` (9 shard files + `model.safetensors.index.json`)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — first run succeeded.
- Pitfalls or surprises you hit (one line each):
  - The two embedding tensors are float16 206 MB in the input but become
    float32 412 MB after the required upcast, so each ends up alone in its
    own shard (well over the 256 MiB budget) per the "single oversized
    tensor gets its own shard" rule — worth double-checking against the
    *output* dtype size, not the input size, when reasoning about shard
    placement.
  - The regex for the 64 bfloat16 targets and the regex for the 48 dropped
    buffers need to be mutually exclusive and jointly exhaustive with
    "everything else is float32"; wrote them as two independent patterns
    and asserted every other tensor is float32 to catch any gap.
- Anything in the task text or documentation that was unclear: none; the
  per-layer tensor names and shapes in TASK.md were sufficient to write
  exact regexes without loading the checkpoint interactively first.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3 — direct `safe_open`/`save_file` for reading the
    single input file and writing sharded output with an index; this task
    is a precise per-tensor dtype/drop/shard rule, which a plain script on
    top of the low-level library gives full control over without adapting
    a higher-level tool's own sharding or naming conventions
    (`transformers`/`mergekit` dtype-export paths save whole models via
    `save_pretrained`, which is a worse fit for one raw `.safetensors`
    input with no model class attached).
  - `torch` 2.14.0 — `.to(torch.bfloat16)` / `.to(torch.float32)` casts.
- Approximate time spent, if you can tell: ~10 minutes.
