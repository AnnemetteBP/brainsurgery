# T5 participant self-report (condition B, pythia-1b)

- Final artifact path: `out/T5/plan.yaml`; output checkpoint in `out/T5/`
  (`model-0000{1..4}-of-00004.safetensors` + `model.safetensors.index.json`).

- Number of times you executed the script or plan: 1
  (plus one throwaway 2-tensor probe plan under `out/_probe/`, since deleted,
  that deliberately failed an assert to confirm a failing check aborts the run
  with exit code 1; it did not touch `out/T5`.)

- Which executions failed, and why (one line each):
  - None. The single execution of `out/T5/plan.yaml` passed all asserts and
    wrote the sharded output.

- Pitfalls or surprises you hit (one line each):
  - dtype mixing: the adapter is float32 and the base is float16, so `add_`
    could not be applied directly; I lifted the 16 target weights with `cast_`
    to float32, added the delta, then `cast_` back to float16.
  - Output-alias inference with two inputs: every destination (including the
    `matmul` result and the temporaries) had to be written into the `base::`
    alias, otherwise the run would fail with `cannot infer output model
    uniquely`; the `lora::` alias is then simply never written out, which also
    guarantees no `lora_*` tensor reaches the output.
  - Intermediates live in the same state dict as the model, so the 16 delta
    tensors had to be explicitly `delete`d before the output was written; I
    gave them a distinctive `bs_tmp.` prefix so a single regex deletes them and
    a `not: exists` assert can prove they are gone.
  - PEFT name prefix: adapter keys carry `base_model.model.` in front of the
    base name, so the regex capture on the layer index is what pairs A, B and
    the base weight (`\1` rewrite in `from_b` and in the `to` destination).
  - Shard budget units: `512MB` in the plan is binary (512 MiB = 536,870,912
    bytes of tensor data), which is exactly the budget the task asks for; the
    four shards came out at 505.1 / 500.3 / 500.3 / 488.2 MiB of tensor data.
  - The task text says `gpt_neox.embed_in.weight` and `embed_out.weight` are
    "larger than that" and get their own shard, but at 206 MB each they fit
    inside the 512 MiB budget and were packed with other tensors; the
    single-tensor-alone rule simply never triggers for this checkpoint.

- Anything in the task text or documentation that was unclear:
  - The embed tensor remark above reads as a statement of fact about this
    checkpoint but is really a hypothetical about the packing rule.
  - The docs do not spell out what `add_` does on mismatched dtypes, so I
    avoided the situation rather than testing it.
  - The ternary transforms' capture/rewrite model (`from_a` drives, `from_b`
    and `to` are rewrites) is only stated in the interfaces reference, not in
    the `matmul` help text.

- Tools used (condition F): n/a (condition B).

- Approximate time spent, if you can tell: ~10 minutes, of which the plan run
  itself was ~9 seconds.

## What the plan does

For each layer `i` in 0..15, with `A = lora_A.weight`, `B = lora_B.weight`,
`scale = lora_alpha / r = 32 / 16 = 2` and `fan_in_fan_out = false`:

1. `matmul` `B @ A` (float32) into `base::bs_tmp.delta.<i>.weight`;
2. `scale_` that delta by 2.0;
3. `cast_` `gpt_neox.layers.<i>.attention.query_key_value.weight` to float32,
   `add_` the delta, `cast_` back to float16;
4. `delete` all `bs_tmp.*` intermediates;
5. write `out/T5` as safetensors with `shard: 512MB`.

Required checks, all as `assert` transforms in the plan:

- exactly 16 adapter pairs found and merged: `count` of the `lora_A` refs,
  of the `lora_B` refs and of the 16 resulting `bs_tmp.delta.*` products;
- no `lora_` tensor in the output: `not: { exists: 'base::.*lora_.*' }`
  (plus `not: { exists: 'base::bs_tmp\..*' }` for the intermediates);
- `gpt_neox.layers.0.attention.query_key_value.weight` has shape
  `[6144, 2048]` (asserted for layer 0 and for all 16 layers), dtype float16;
- the output has exactly 244 tensors: `count: { of: 'base::.*', is: 244 }`.
