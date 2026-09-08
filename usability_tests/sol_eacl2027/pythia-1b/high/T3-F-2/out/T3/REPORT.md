## Participant self-report

- Final artifact path: `out/T3/`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): None; the first execution succeeded.
- Pitfalls or surprises you hit (one line each): The embeddings are about 206 MB in the float16 input but 412,090,368 bytes (about 393 MiB) after the required float32 upcast, so each needs the oversized-single-tensor shard exception.
- Anything in the task text or documentation that was unclear: The statement that the embeddings are 206 MB each appears to describe their input float16 size, while the required output float32 tensors are about 412 MB each; the singleton exception still makes the intended result unambiguous.
- Tools used (condition F): `torch` 2.14.0 for exact dtype conversion and equality checks; `safetensors` 0.5.3 for lazy input reads and sharded checkpoint writes; Python standard library for deterministic shard planning and JSON index generation. This route avoids materializing the entire checkpoint at once.
- Approximate time spent, if you can tell: About 6 minutes.
