## Participant self-report

- Final artifact path: `out/T5/solution.py`
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none, the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The two ~412 MB tensors (`model.embed_tokens.weight`, `lm_head.weight`) are individually
    under the 512 MiB shard budget, so a naive greedy packer wouldn't isolate them on its own;
    isolating any tensor over half the shard budget into its own shard was the simplest rule
    that matches the spec's stated requirement without hardcoding tensor names.
  - Adapter keys are prefixed `base_model.model.` (PEFT convention) versus the base checkpoint's
    plain `model.` prefix; had to strip that prefix to map `lora_A`/`lora_B` pairs onto base
    tensor names.
- Anything in the task text or documentation that was unclear:
  - The "512 MiB" note about the two large tensors reads as if they exceed the shard budget,
    but at 412 MB each they're actually under it; the practical reading (isolate large tensors
    rather than pack them tightly) is what I implemented.
- Tools used (condition F): n/a (condition P).
- Approximate time spent, if you can tell: ~15 minutes.
