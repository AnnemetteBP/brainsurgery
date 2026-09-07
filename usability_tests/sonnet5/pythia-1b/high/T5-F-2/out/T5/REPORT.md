# T5 (condition F, Pythia-1B) — participant self-report

- **Final artifact path:** `out/T5/solution.py` (invoked via `out/T5/run.sh`),
  output checkpoint in `out/T5/` (4 shards + `model.safetensors.index.json`).
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit:**
  - The base checkpoint mixes dtypes (`float16` weights plus `uint8` buffers,
    e.g. `attention.bias` masks); the merge must only touch the 16 float16
    `query_key_value.weight` tensors and pass everything else through
    untouched, including the non-float16 buffers.
  - Adapter tensor names carry a `base_model.model.` prefix (PEFT convention)
    that has to be stripped to align with the base checkpoint's own key
    space; a naive substring match on module name alone would have been
    ambiguous.
- **Anything in the task text or documentation that was unclear:** the task
  says a shard budget of "512 MiB (536,870,912 bytes)" and then, in the same
  sentence, gives `gpt_neox.embed_in.weight` / `embed_out.weight` (206 MB
  each, confirmed by reading the input file) as examples of tensors "larger
  than that" that must be isolated in their own shard. 206 MB is well under
  536,870,912 bytes, so those two tensors do not actually trigger the
  "oversized tensor gets its own shard" fallback under the stated budget —
  the two statements are inconsistent for this target. I implemented the
  literal, well-defined rule (greedy-pack shards up to 512 MiB; a tensor that
  individually exceeds that budget is placed alone) rather than
  hard-coding those two tensor names, since the numeric rule is unambiguous
  and this is the same algorithm `transformers`' own sharded `save_pretrained`
  uses. In this run, `embed_in.weight`/`embed_out.weight` end up bin-packed
  alongside other tensors in shard 1 rather than isolated; every shard still
  respects the 512 MiB cap.
- **Tools used (condition F):** `safetensors` 0.5.3 and `torch` 2.14.0 only,
  via a plain script (`out/T5/solution.py`), not `peft.merge_and_unload`.
  Reasoning: the spec pins down exact arithmetic (`scale = lora_alpha / r`,
  `B @ A` with no transpose since `fan_in_fan_out = false`, float32
  accumulation, cast back to float16) and exact output shape (sharded,
  244 tensors, no adapter/intermediate leftovers). Doing this as direct
  tensor-dict surgery makes every one of those steps and every required
  check (pair count, no `lora_` names, shape probe, tensor count,
  bit-exact pass-through of untouched tensors) an explicit, auditable
  assertion in the script, rather than trusting `peft`'s internal merge
  dtype handling and `transformers`' shard-size heuristics inside
  `save_pretrained`, which would need instantiating the full GPTNeoX model
  and would hide these decisions behind library internals. The script also
  self-verifies bit-exactness of the 228 untouched tensors and the merged
  values by recomputation before writing.
- **Approximate time spent:** ~10 minutes (inspection of inputs, one
  script write, one successful run, independent verification).
