# Participant self-report: T2 (Pythia-1B head pruning), condition B

- Final artifact path: `out/T2/plan.yaml` (output checkpoint `out/T2/model.safetensors`, 244 tensors)
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; execution 1 passed all asserts and wrote the output.
- Pitfalls or surprises you hit (one line each):
  - `concat` requires each source reference to resolve to exactly one tensor, so the 16-layer edit cannot be written as one pattern-based concat; the plan has 48 explicit concat blocks (3 per layer) into a temporary `pruned.` prefix, followed by pattern `delete` of the originals and one regex `move` (`pruned\.(.+)` -> `gpt_neox.\1`) to restore the names.
  - Head 5 in the interleaved GPT-NeoX layout is rows 3840..4607 of `query_key_value` (768 rows per head), and columns 1280..1535 of `dense.weight` (256 per head); the slices `[:3840]` + `[4608:]` and `[:, :1280]` + `[:, 1536:]` were used.
  - Dots in tensor names were escaped in the regex references to avoid accidental matches.
- Anything in the task text or documentation that was unclear:
  - The README does not state that `concat` refuses multi-match sources; only `help.txt` says so. A pattern-based concat (or a `delete`-style slice removal) would have made this a 3-transform plan.
  - The `concat` help shows an example with an empty `from:` list, which looks like a rendering glitch.
- Tools used (condition F): n/a (condition B, BrainSurgery plan only)
- Approximate time spent, if you can tell: about 3 minutes (reading the doc pack, writing the plan, one run of ~10 s).
