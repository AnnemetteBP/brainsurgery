# T5 — Participant self-report

- **Final artifact path:** `out/T5/solution.py` (runner: `out/T5/run.sh`,
  invoked from the sandbox root as `sh out/T5/run.sh`).

- **Number of times you executed the script or plan:** 2. Both produced the
  correct output and passed every check; the second was needed because the
  first destroyed its own source file (see below). (I also ran a separate
  read-only verification snippet that re-derives the merge in float64 and
  compares against `out/T5/`; it writes nothing and is not an attempt.)

- **Which executions failed, and why:** Neither failed a check or exited
  non-zero. Execution 1 nonetheless had a real defect: its pre-write cleanup
  was `shutil.rmtree(OUT_DIR)`, and since the task requires the artifact to
  live in `out/T5/` too, it deleted `solution.py` and `run.sh` on its way to
  writing the shards. The checkpoint it produced was valid — Python had
  already read the source — but the artifact was gone. I rewrote the cleanup to
  unlink only `*.safetensors` and the index file, and re-ran (execution 2).

- **Pitfalls or surprises you hit:**
  - The obvious condition-F route — `peft.merge_and_unload` + `save_pretrained`
    — is wrong for this task: the base checkpoint stores buffers that
    `GPTNeoXForCausalLM` treats as non-persistent (`attention.bias` as U8
    `[1,1,2048,2048]`, `attention.masked_bias`, `rotary_emb.inv_freq`, 48
    tensors in total). Round-tripping through a model instance would emit ~196
    tensors, not the required 244. I checked the base key set before choosing
    an approach, which is what caught this.
  - PEFT name prefix: adapter keys carry `base_model.model.` in front of the
    base name and `.lora_A/.lora_B` behind it; the base name is what is left
    after stripping both. I matched with an anchored regex rather than a
    substring replace so an unexpected name fails loudly instead of silently
    not matching.
  - `attention.bias` is `uint8`, so tensor byte sizes are not `numel * 2`
    across the board; the shard budget has to be computed from
    `numel * element_size()`, not from the shape alone.
  - Writing the artifact into the same directory as the output makes a
    wholesale `rmtree` of the output directory self-destructive. Scoping the
    cleanup to the file types the run actually produces is the fix; deleting a
    directory that is also holding your source is not recoverable from inside
    the same run.
  - The task text says `gpt_neox.embed_in.weight` and `embed_out.weight`
    ("206 MB each") are "larger than" the 512 MiB budget and must sit alone.
    They are not: each is 50304x2048 fp16 = 206 MB = 196.5 MiB, comfortably
    under 536,870,912 bytes. I implemented the rule as stated (a tensor over
    the budget is allowed to sit alone in its own shard) and asserted the
    invariant; here the clause is simply vacuous and the greedy packing puts
    the two embeddings in shards with other tensors. Result: 4 shards of
    505.1 / 500.3 / 500.3 / 488.2 MiB of tensor data.

- **Anything in the task text or documentation that was unclear:**
  - The "206 MB > 512 MiB" claim above is the one contradiction; it reads like
    it was inherited from a smaller target. It left me unsure whether the
    hidden reference forces the embeddings into single-tensor shards. I went
    with the numeric rule (≤ 512 MiB per shard) rather than the parenthetical.
  - "Sharding rules" in the grading section does not say whether shard *file
    assignment* must match the reference or only the constraints. I used
    `huggingface_hub.split_torch_state_dict_into_shards` with the standard
    `model-0000k-of-0000n.safetensors` pattern, i.e. exactly what
    `transformers.save_pretrained` would produce, as the most likely
    convention.

- **Tools used (condition F):**
  - `safetensors` 0.5.3 — `load_file` / `save_file` / `safe_open` for direct
    tensor-level read and write. Chosen because the task is a checkpoint edit,
    not a model edit: it preserves the exact key set, dtypes and bit patterns
    of the 228 untouched tensors, which a model round-trip does not.
  - `torch` 2.14.0 — the `B @ A` matmul in float32 and the cast back to
    float16, plus `torch.equal` for the bit-exactness audit.
  - `huggingface_hub` (pinned, via `split_torch_state_dict_into_shards`) — the
    same splitter `transformers.save_pretrained` uses, so the shard layout,
    file naming and `model.safetensors.index.json` schema follow the standard
    convention instead of a hand-rolled one.
  - `peft` 0.20.0 — **not** used for the merge (see pitfalls); I only read
    `adapter_config.json` directly for `r`, `lora_alpha` and `fan_in_fan_out`.
  - `transformers`, `mergekit`, `torch-state-bridge` — not used. mergekit's
    task arithmetic works on full fine-tuned models, not low-rank factors, and
    key rewriting was not needed since the name mapping is a fixed prefix/suffix
    strip.

- **Approximate time spent:** about 8 minutes.

## Checks enforced by the run

All of these run before anything is written, and raise `CheckFailed`
(non-zero exit) if they do not hold:

- exactly 16 complete adapter pairs found, and exactly 16 weights merged;
- `fan_in_fan_out` is false and each factor's rank matches `r = 16` (an
  unhandled layout aborts instead of silently transposing);
- each target's base shape equals the `B @ A` shape;
- no output tensor name contains `lora_`;
- `gpt_neox.layers.0.attention.query_key_value.weight` has shape `[6144, 2048]`
  and dtype float16;
- the output has exactly 244 tensors, with the same key set as the base;
- the 228 non-adapted tensors are bit-identical to the base
  (`torch.equal` against the input file);
- every shard holds at most 536,870,912 bytes of tensor data, unless it holds a
  single over-budget tensor; the split covers each tensor exactly once.

After writing, the run reloads `out/T5/` through the index and re-checks the
tensor count, absence of `lora_`, the probe shape, per-tensor dtype/shape,
bit-equality with the in-memory result, and that the set of `*.safetensors`
files on disk is exactly the set the index references (so a stale shard from an
earlier run cannot survive unnoticed); any mismatch is an error.

## Result of the run

```
adapter: 16 pairs, scale = alpha/r = 2.0
merged 16 weights; 228 tensors untouched
wrote 244 tensors into 4 shards
  model-00001-of-00004.safetensors: 529608772 bytes (505.1 MiB)   [ 24 tensors]
  model-00002-of-00004.safetensors: 524554570 bytes (500.3 MiB)   [ 75 tensors]
  model-00003-of-00004.safetensors: 524554570 bytes (500.3 MiB)   [ 75 tensors]
  model-00004-of-00004.safetensors: 511955272 bytes (488.2 MiB)   [ 70 tensors]
OK
```

Independent post-hoc verification (float64 recomputation of
`base + 2 * B @ A`): worst relative Frobenius error over the 16 merged
weights is 2.076e-04, against the 1e-3 limit.

The 228 unchanged tensors are bit-identical to the base, the key set matches
the base exactly (244 names), and no shard exceeds 536,870,912 bytes of tensor
data.
