## Participant self-report

- Final artifact path: `out/T2/solution.py` (invoked via `out/T2/run.sh`), output at `out/T2/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none.
- Pitfalls or surprises you hit (one line each): none — the task spec gave exact row/column ranges to keep, so this was a direct slice-and-concat with no ambiguity about which axis holds the heads for each projection.
- Anything in the task text or documentation that was unclear: no.
- Tools used (condition F): name, version, and why: `torch` 2.14.0 and `safetensors` 0.5.3 only, via a plain Python script — loaded the sharded checkpoint with `safetensors.torch.load_file`, sliced `q_proj`/`k_proj`/`v_proj` (row blocks) and `o_proj` (column blocks) around the head-5 block per layer with `torch.cat`, left all other tensors untouched, asserted the required shapes and the 114-tensor count, then wrote a single `safetensors.torch.save_file`. Did not use `transformers.prune_heads` (it operates on a loaded model and reindexes/prunes heads by index sets in a way I'd have had to verify matched the exact keep-order semantics in the spec) or `mergekit`/`torch-state-bridge` (built for cross-checkpoint merging/renaming, not intra-tensor slicing) — a direct script gave full, auditable control over the exact slice boundaries and concatenation order the task specifies.
- Approximate time spent, if you can tell: a few minutes.
