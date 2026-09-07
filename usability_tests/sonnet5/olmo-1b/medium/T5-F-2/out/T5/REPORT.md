# T5 report (condition F)

- Final artifact path: `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `crash`: `AssertionError: base checkpoint is missing model.model.layers.0.self_attn.q_proj.weight` — built the base key by prepending `model.` to the adapter's module path, but the adapter's `base_model.model.` prefix strips down to a path that already starts with `model.layers...`, so the prefix was duplicated.
- Pitfalls or surprises you hit (one line each):
  - The adapter key prefix is `base_model.model.` (PEFT wraps the HF model in `base_model.model`), which after stripping still leaves the full `model.layers.<i>...` path, not a bare `layers.<i>...` path.
  - `model.embed_tokens.weight` and `lm_head.weight` (412 MB each) are individually under the 512 MiB shard cap, so a size check of `nbytes > 512 MiB` alone would pack them with a neighbor; had to add an explicit "isolate anything past half a shard" rule to match the task's stated requirement that they land alone in their own shard.
- Anything in the task text or documentation that was unclear:
  - The 512 MiB spec technically only forces a tensor "larger than that" to be alone, but 412 MB is smaller than 512 MiB; the task also states as fact that these two tensors end up alone, so the two statements only reconcile if "isolate" is a bin-packing outcome (any other tensor added would push the shard's total past a reasonable slot) rather than a hard per-tensor size rule. I used a half-shard-size threshold to reproduce the stated result.
- Tools used (condition F): name, version, and why:
  - `torch` 2.14.0 — tensor math (`B @ A`) and dtype/shape handling.
  - `safetensors` 0.5.3 (`safe_open`, `save_file`) — reading the sharded base checkpoint and the adapter file directly, and writing the merged sharded output. Chose a direct safetensors script over `peft.merge_and_unload` because that path requires instantiating the full HF model (extra dependency surface and memory) for a transform that is purely a per-tensor rename + multiply + add; a plain script gives exact control over the output sharding rule (512 MiB cap, oversized tensors alone) that the task specifies precisely, which `merge_and_unload` plus a generic HF `save_pretrained(..., max_shard_size=...)` call would approximate but not guarantee bit-for-bit against the stated rule for the two large tensors.
  - `json` (stdlib) — reading `adapter_config.json` / `model.safetensors.index.json` and writing the output index.
- Approximate time spent, if you can tell: ~10 minutes (one script draft, one bug fix, verification against a manually recomputed reference).
