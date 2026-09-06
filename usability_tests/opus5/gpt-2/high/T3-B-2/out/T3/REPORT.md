# T3 self-report (Condition B, GPT-2 124M)

- **Final artifact path:** `out/T3/plan.yaml` (output checkpoint in `out/T3/`:
  `model-00001..00004-of-00004.safetensors` + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  - Execution 1 — `failed_assertion` / `no_match`: `TransformError: count.of matched zero tensors: model::h\.\d+\.attn\.bias`. I had written the "buffers are gone" check as `count: { of: 'h\.\d+\.attn\.bias', is: 0 }`, but the reference resolver raises on a zero-match reference before `count` ever compares, so a count of 0 is not expressible. Replaced with `not: { exists: 'h\.\d+\.attn\.bias' }`. Every earlier transform and check in the plan had already passed at that point; nothing was written (the failure is before the output stage, so no partial output).
  - Execution 2 — passed.

- **Pitfalls or surprises you hit (one line each):**
  - `count: is: 0` cannot express absence; the resolver errors out on a reference that matches nothing, so absence has to go through `not: { exists: ... }`.
  - `assert.exists` takes a bare tensor-ref as its payload, not an `{ of: ... }` mapping like `count`/`dtype`/`shape` do — the asymmetry is only visible in `help.txt`'s "Payload:" line.
  - The buffer to delete is `h.<i>.attn.bias`, which sits in the same namespace as the projection biases `h.<i>.attn.c_attn.bias` / `c_proj.bias` that must survive; `h\.\d+\.attn\.bias` as a full-match regex separates them cleanly, but a sloppy `h.*attn.*bias` would take all of them.
  - The stated hazard is real: `.*weight` would have hit `wte.weight`, `wpe.weight` and 48 layer-norm weights. I anchored the cast on the four module names instead and asserted the complement (100 tensors) is still float32, which is what actually pins down "exactly 48 bfloat16" — `count` alone counts name matches, not dtypes.
  - Shard budget units are binary (`64MB` = 67,108,864 bytes), and the budget counts tensor bytes only. `wte.weight` (154 MB) exceeded it and the writer put it alone in shard 4 without any special handling, as documented.

- **Anything in the task text or documentation that was unclear:**
  - "exactly 48 tensors are bfloat16" is a check over dtypes, but no assert operator selects tensors *by* dtype — `count` matches names and `dtype` tests matches. I expressed it as three asserts (48 name matches + those are bf16 + the remaining 100 are f32); a `count` with a dtype filter would say it directly.
  - The README documents `count` for "exact number of matches" without noting that zero is not a legal expectation.

- **Tools used (condition F):** n/a (condition B).

- **Approximate time spent, if you can tell:** ~10 minutes.

## Verification performed

Independently re-read the written shards (not via brainsurgery) and confirmed:
148 tensors, `weight_map` covers exactly those 148 and names exactly the 4 shard
files present; shard tensor bytes 66,259,968 / 66,256,896 / 40,983,552 (all
within 67,108,864) and 154,389,504 for `wte.weight` alone in its own shard;
exactly 48 bfloat16 tensors and they are exactly the 48 projection matrices;
every tensor bit-exact against `input.to(bfloat16)` for the projections and
against the unchanged float32 input for the other 100; the 12 deleted keys are
exactly `h.<i>.attn.bias`; no renamed or extra keys.
