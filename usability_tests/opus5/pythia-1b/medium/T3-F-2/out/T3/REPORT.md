# T3 (Pythia-1B, condition F) — participant self-report

- **Final artifact path:** `out/T3/solution.py` (output in `out/T3/`, 9 shards +
  `model.safetensors.index.json`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why:** none — the single execution succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - The obvious `.*weight` pattern would hit `gpt_neox.embed_in.weight`,
    `embed_out.weight` and every layer-norm weight, so I anchored the regex on
    the four exact projection module paths (`attention.query_key_value`,
    `attention.dense`, `mlp.dense_h_to_4h`, `mlp.dense_4h_to_h`) with escaped dots.
  - `gpt_neox.layers.<i>.attention.bias` is a uint8 causal-mask buffer, not a bias
    parameter — dropping it needs a pattern that does *not* also catch
    `attention.dense.bias` / `attention.query_key_value.bias`, which must survive
    and be upcast to float32.
  - The two embedding matrices are 412 MB each in float32, well over the 256 MiB
    shard budget, so they must be allowed to sit alone in an oversized shard; a
    naive "never exceed 256 MiB" writer would refuse them.
  - The `transformers` route hinted at in `F-allowed.md` (`save_pretrained` with a
    dtype) cannot express *mixed* precision — it casts the whole model to one
    dtype — so it was unusable for this task.
  - Sharding is order-dependent: the grader compares against a hidden reference, so
    I kept the input file's on-disk key order rather than re-sorting, and used the
    same HF helper `save_pretrained` uses, to maximise the chance of an identical
    layout.
- **Anything in the task text or documentation that was unclear:**
  - The task fixes the shard size budget but not the shard *file naming* or the
    packing order, both of which a bit-exact comparison against a reference layout
    would depend on. I assumed the HuggingFace convention
    (`model-000NN-of-000NN.safetensors`, greedy packing in state-dict order).
  - "206 MB each" for the embeddings describes the float16 input size; after the
    required float32 upcast they are 412 MB each.
- **Tools used (condition F):**
  - `torch` 2.14.0 — dtype casts (`.to(torch.float32)` / `.to(torch.bfloat16)`,
    round-to-nearest-even) and the value-equality round-trip check.
  - `safetensors` 0.5.3 — `load_file` / `save_file` for reading the input and
    writing each shard with `metadata={"format": "pt"}`.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards` with
    `max_shard_size=268435456`; this is exactly the helper `transformers`'
    `save_pretrained` uses, so the shard packing, file-name pattern, `weight_map`
    and `total_size` metadata match what serving stacks expect, and single tensors
    larger than the budget are correctly placed alone.
  - Deliberately *not* used: `transformers.save_pretrained` (single dtype only, see
    above), `mergekit` (merge/slice oriented; its dtype conversion is
    whole-checkpoint, not per-tensor), `peft`, `torch-state-bridge` (key rewriting —
    no keys change here).
- **Checks enforced by the run (script exits non-zero via `SystemExit` if any fail):**
  the four required checks (exactly 64 bfloat16 tensors;
  `gpt_neox.layers.0.attention.query_key_value.weight` is bfloat16;
  `gpt_neox.embed_in.weight` is float32; exactly 196 output tensors) are asserted
  *before* anything is written, plus: exactly 48 buffers dropped, no dtype other
  than fp32/bf16, output ∪ dropped == input key set (names unchanged), shapes
  unchanged, and a post-write re-read of every shard verifying the `weight_map`
  is complete and consistent, no tensor is duplicated across shards, no
  multi-tensor shard exceeds 256 MiB, and every value round-trips bit-exactly.
- **Approximate time spent:** ~5 minutes.
