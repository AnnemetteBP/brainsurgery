# Participant self-report

- Final artifact path: `out/T5/model.safetensors.index.json` and `out/T5/model-00001-of-00010.safetensors` through `out/T5/model-00010-of-00010.safetensors` (producer: `out/T5/solution.py`; independent checker: `out/T5/verify.py`).
- Number of times you executed the script or plan: 1 production execution.
- Which executions failed, and why (one line each): None; the first production execution succeeded.
- Pitfalls or surprises you hit (one line each): The two approximately 412 MB embedding/head tensors are smaller than 512 MiB despite being described as larger than the limit, so I followed the explicit requirement and isolated each in its own shard.
- Anything in the task text or documentation that was unclear: Only the 412 MB versus 512 MiB wording noted above; the required layout, scaling, naming, checks, and output location were otherwise clear.
- Tools used (condition F): `torch` 2.14.0 for float32 `B @ A` and tensor comparisons; `safetensors` 0.5.3 for header inspection and streaming checkpoint reads/writes; Python standard-library JSON/path utilities for index generation and validation. Direct checkpoint processing was chosen to avoid instantiating the model.
- Approximate time spent, if you can tell: About 8 minutes.
- Artifact LOC: 165 non-blank, non-comment lines in `out/T5/solution.py`.

Independent read-back verification passed: all 114 names, shapes, and float32 dtypes are present; all 32 merged tensors exactly match recomputation; all 82 unchanged tensors are bit-identical to the base; both large tensors are isolated; and every shard contains at most 536,870,912 bytes of tensor data.
