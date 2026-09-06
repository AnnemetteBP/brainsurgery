# T5 report (condition F)

## Participant self-report

- Final artifact path: `out/T5/solution.py` (invoked via `out/T5/run.sh`)
- Number of times you executed the script or plan: 2
- Which executions failed, and why (one line each):
  1. `crash`: `KeyError`-style `AssertionError` — I first looked up the base
     tensor by the adapter's raw capture group (`h.0.attn.c_attn`) without
     appending `.weight`, so the base-name lookup failed for every pair.
- Pitfalls or surprises you hit (one line each):
  - Adapter keys are `base_model.model.<module>.lora_A/B.weight`; the module
    name has to gain a trailing `.weight` to match the base checkpoint's
    naming, it's not a 1:1 string match.
  - Had to keep the Conv1D `[in, out]` vs `nn.Linear` `[out, in]` distinction
    explicit: `B @ A` is `[out, in]`, so it needs `.T` before adding to the
    `[in, out]` base weight, which is exactly what `fan_in_fan_out=True`
    signals.
  - The 100 MiB shard budget is on tensor bytes only (header excluded), and
    `wte.weight` alone is ~154 MB, so the greedy packer has to allow a shard
    with a single oversized tensor rather than trying to always fill to the
    budget.
- Anything in the task text or documentation that was unclear: No — the
  formula, layout, and shard budget were all stated precisely enough to
  implement directly.
- Tools used (condition F): `torch` 2.14.0 and `safetensors` 0.5.3 only, as a
  plain script — no `peft.merge_and_unload` or `mergekit`. The task never
  requires instantiating a `GPT2LMHeadModel`, and doing the merge on the raw
  tensors is simpler and easier to shard and check than round-tripping
  through PEFT's model wrapper and `save_pretrained`'s own sharding, which
  offers less direct control over the "single oversized tensor gets its own
  shard" rule.
- Approximate time spent, if you can tell: a few minutes of scripting plus
  one quick fix-and-rerun cycle.
