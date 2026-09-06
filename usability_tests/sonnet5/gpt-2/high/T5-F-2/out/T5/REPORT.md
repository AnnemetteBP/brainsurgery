# T5 participant self-report

- **Final artifact path:** `out/T5/solution.py` (run via `out/T5/run.sh`), output
  checkpoint at `out/T5/model-*-of-00005.safetensors` + `out/T5/model.safetensors.index.json`.
- **Number of times you executed the script or plan:** 2.
- **Which executions failed, and why (one line each):**
  1. `crash` — `KeyError`-style `SystemExit`: I stripped the `.lora_A.weight`/
     `.lora_B.weight` suffix from adapter keys to get the base tensor name but
     forgot the adapter names the *module* (`h.<i>.attn.c_attn`) while the base
     checkpoint stores the *tensor* (`h.<i>.attn.c_attn.weight`); looked up the
     wrong key and the "adapter targets X, which is not in the base checkpoint"
     guard fired as designed.
- **Pitfalls or surprises you hit (one line each):**
  - Adapter keys name the target *module*, not the target *tensor*; needed to
    append `.weight` before indexing into the base state dict.
  - The base checkpoint is a bare state dict (`h.0...`, `wte.weight`, no
    `transformer.` prefix), i.e. a raw `GPT2Model` dump rather than a full
    `GPT2LMHeadModel` — going through `transformers`/`peft`'s
    `from_pretrained` + `merge_and_unload()` would have required reconciling
    that prefix mismatch and then re-deriving custom shard packing anyway to
    hit the exact 100 MiB tensor-data budget, so a direct safetensors+torch
    script was more predictable and easier to make fail loudly on the
    required checks.
  - The 100 MiB shard budget is tensor-data bytes only, not file size —
    doesn't matter much here since safetensors headers are small, but worth
    computing shard sizes from tensor `nbytes` rather than the resulting file
    size to avoid conflating the two.
  - `(B @ A).T` is non-contiguous; called `.contiguous()` before merging so
    `save_file` doesn't complain about non-contiguous/shared storage.
- **Anything in the task text or documentation that was unclear:** No —
  the fan_in_fan_out / Conv1D-vs-Linear layout explanation and the shard
  budget rule (oversized tensor gets its own shard) were both stated
  explicitly and matched what I implemented.
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — direct load/save of both checkpoints and the
    sharded output; gives full control over which tensors go in the output
    and how shards are packed, which the exact-160-tensors and
    exact-100-MiB-budget requirements needed.
  - `torch` 2.14.0 — tensor ops for the merge math (`B @ A`, transpose,
    scale, add) in float32.
  - Did **not** use `peft.merge_and_unload()`: it operates on an
    instantiated `nn.Module`, and this base checkpoint's key layout (bare
    `GPT2Model` state dict) doesn't map cleanly onto a stock HF model class
    without extra prefix bookkeeping; a plain script on the checkpoint files
    was more direct and easier to guard with explicit assertions.
- **Approximate time spent, if you can tell:** ~10 minutes end to end
  (inspecting inputs, writing the script, one failed run, one fix, verifying
  merged values against an independent reimplementation of the formula).
