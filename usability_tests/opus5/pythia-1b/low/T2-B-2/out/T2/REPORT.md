# T2 self-report (condition B, Pythia-1B)

- **Final artifact path:** `out/T2/plan.yaml` -> `out/T2/model.safetensors`
- **Number of times you executed the script or plan:** 2
- **Which executions failed, and why (one line each):**
  - Execution 1: `no_match` — my extra sanity check `assert: count: { of: 'tmp\..*', is: 0 }` errored with "count.of matched zero tensors" because a reference that matches nothing is an error before `count` is ever evaluated; the surgery itself had already succeeded. Removed that assert.
- **Pitfalls or surprises you hit (one line each):**
  - `count` (and every target reference) cannot be used to assert absence: zero matches raises rather than returning 0.
  - `concat` requires each `from` entry to resolve to exactly one tensor, so it cannot be batched with captures over the 16 layers; I generated the 16x3 copy/delete/concat groups as explicit plan entries.
  - Destinations must not already exist, so the pattern is: copy the two keep-slices to scratch names, `delete` the original, `concat` the scratch pieces back onto the original name, then `delete` the scratch names.
  - The GPT-NeoX interleaved qkv layout means head 5 is a single contiguous 768-row block (3840..4607), so the "remove one head" edit is one contiguous cut on rows, and a 256-wide column cut (1280..1535) on `attention.dense.weight` — no per-q/k/v regrouping needed.
  - Layer-index regexes like `layers.0.attention...` are full-match, so `0` does not overmatch `10`.
- **Anything in the task text or documentation that was unclear:** Task text was precise about the layout. The docs do not state that a zero-match reference is a hard error even for `count`, which is what cost me the one failed execution.
- **Tools used (condition F):** n/a (condition B).
- **Approximate time spent, if you can tell:** ~5 minutes.

## Verification performed

Beyond the in-plan asserts (shapes of the three layer-0 tensors and total tensor count 244), I independently reloaded input and output and confirmed: identical key set (244), the three head-bearing tensors per layer equal the expected index-select of the input, every other tensor bit-identical, dtype float16 preserved.
