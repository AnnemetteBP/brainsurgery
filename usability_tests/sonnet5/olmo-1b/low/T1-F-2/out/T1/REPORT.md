# T1 participant self-report

## Tools used

Plain Python script on top of the `safetensors` library only (`safe_open` /
`save_file`), no `mergekit` or `torch-state-bridge`.

**Why:** the task is a single, well-specified rename/drop/repack over a flat
key namespace (`model.layers.<i>.<rest>`). Writing the mapping directly
(regex-match the layer index, drop the four target blocks, remap survivors
through an explicit `old -> new` dict built from list order) is fewer moving
parts than authoring a mergekit YAML slice config or reaching for
torch-state-bridge's rule engine, and it's easiest to make the collision and
count checks explicit and load-bearing rather than trusting a general tool's
defaults.

## What the solution does

1. Loads all tensors from both input shards via the index's `weight_map`
   (fails if loaded keys don't match the index).
2. For each tensor: non-block tensors pass through unchanged; block tensors
   for indices in `{2, 6, 10, 14}` are dropped; the rest are renamed using an
   explicit `old_idx -> new_idx` mapping derived by enumerating the sorted
   surviving indices, preserving original order (0,1,3,4,5,7,8,9,11,12,13,15
   -> 0..11).
3. Detects and raises on any key collision during renaming.
4. Runs the required checks before writing anything: no leftover
   blocks 12-15, exactly 12 distinct block indices (0..11), exactly 12
   `q_proj` tensors, non-block tensor set is exactly
   `{embed_tokens.weight, lm_head.weight}`, and exactly 86 tensors total.
5. Only if all checks pass, writes `out/T1/model.safetensors` via
   `safetensors.torch.save_file`. On any exception, prints the error to
   stderr, deletes any partial output, and exits non-zero.

## Verification

Ran `python solution.py` once; succeeded on the first attempt. Independently
reopened the output file and confirmed 86 tensors total and block indices
exactly `[0..11]`.

## Executions

- 1 execution, 1 success, 0 failures.
