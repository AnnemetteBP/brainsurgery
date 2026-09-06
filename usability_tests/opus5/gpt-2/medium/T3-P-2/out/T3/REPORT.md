# T3 — Participant self-report (condition P)

- **Final artifact path:** `out/T3/solution.py`; output checkpoint in `out/T3/`
  (`model-0000{1..4}-of-00004.safetensors` + `model.safetensors.index.json`).
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single
  execution succeeded and all required checks passed.
- **Pitfalls or surprises you hit (one line each):**
  - The obvious `.*weight` pattern is a trap, so I never used a regex: I built
    the 48 bfloat16 names explicitly from the four projection templates over
    layers 0..11, which also makes over-matching impossible by construction.
  - `h.<i>.attn.bias` is a causal-mask buffer, not a bias parameter, so name
    matching on `bias` would have deleted 12 real parameters; the drop set is
    likewise an explicit 12-name set.
  - safetensors rejects tensors sharing storage, so every output tensor is
    `.contiguous().clone()`d before saving.
  - Key order from the file puts `wte.weight` last; greedy packing therefore
    lands it alone in the final shard, which is exactly what the 64 MiB rule
    requires for an over-budget single tensor — no special case needed, but I
    kept the budget assertion scoped to multi-tensor shards so a lone oversized
    tensor does not trip it.
  - Shard *file* sizes slightly exceed the tensor-data totals (headers); the
    budget is checked against tensor data only, as the task specifies.
- **Anything in the task text or documentation that was unclear:**
  - The shard *naming* scheme and the packing order are not specified. I used
    the HF convention `model-{i:05d}-of-{n:05d}.safetensors` with greedy
    first-fit packing in the input file's key order, which is what
    `save_pretrained` produces.
  - Whether `metadata.total_size` is required in the index is not stated; I
    included it, again following the HF convention.
- **Tools used (condition F):** n/a — condition P (torch 2.14.0,
  safetensors 0.5.3).
- **Approximate time spent, if you can tell:** a few minutes; one read of the
  task, one script, one run.
