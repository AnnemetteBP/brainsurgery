# T5 participant self-report

- Final artifact path: `out/T5/solution.py` (invoked via `out/T5/run.sh`),
  output checkpoint written to `out/T5/` (`model-0000X-of-00004.safetensors`
  + `model.safetensors.index.json`).
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none — the single
  execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - The task text calls out `gpt_neox.embed_in.weight` / `embed_out.weight`
    (196.5 MiB, "~206 MB") as tensors that must sit alone in their own
    shard, but 196.5 MiB is under the 512 MiB budget, so a plain greedy
    bin-pack correctly folds them in with neighboring tensors instead of
    isolating them; no tensor in this checkpoint actually exceeds 512 MiB,
    so the "alone in its own shard" branch of the sharding logic never
    triggers here (it's still implemented, just unexercised).
  - `lora_A`/`lora_B` keys are prefixed with `base_model.model.`, which has
    to be stripped before matching against base checkpoint key names.
- Anything in the task text or documentation that was unclear: the
  embed tensor size stated in the task (206 MB) versus the 512 MiB shard
  threshold made it look like those two tensors were meant to force their
  own shard; measuring them directly showed they are well under the limit,
  so I trusted the stated numeric threshold over the parenthetical example.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, in
  a plain script — no `peft`/`mergekit`. The merge itself is a two-line
  `float32` matmul-and-add per adapter pair, and the required sharding rule
  (hard 512 MiB cap, oversized tensors alone) is simple enough to implement
  directly with full control over shard boundaries and the index file;
  routing through `peft.merge_and_unload` would have meant instantiating
  the full GPT-NeoX model just to get back a state dict, then handing
  sharding off to `transformers.save_pretrained`, which is less direct and
  gives less certainty about matching the exact bin-packing rule specified
  here.
- Approximate time spent, if you can tell: a few minutes of scripting plus
  one run and a verification pass.
