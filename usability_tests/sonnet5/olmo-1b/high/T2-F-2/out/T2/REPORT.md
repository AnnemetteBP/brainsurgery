# T2 participant self-report

- Final artifact path: `out/T2/solution.py` (invoked via `out/T2/run.sh`), output at `out/T2/model.safetensors`.
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the single execution succeeded.
- Pitfalls or surprises you hit (one line each):
  - None of real consequence. The only thing worth double checking was the head-axis
    direction: q/k/v prune on rows (dim 0, since `nn.Linear` weights are `[out, in]`
    and the head axis is the output/row axis), o_proj prunes on columns (dim 1, the
    input axis, since o_proj consumes the concatenated per-head values). Getting this
    backwards on any one projection would silently produce a shape that still "looks"
    plausible (e.g. slicing o_proj by rows gives `[1920, 2048]` instead of the required
    `[2048, 1920]`), so I added an explicit expected-shape assertion per projection type
    rather than trusting the slice direction by inspection.
- Anything in the task text or documentation that was unclear: no; the row/column
  ranges to keep were given explicitly (`0..639`, `768..2047`), which removed any
  ambiguity about where head 5's 128-wide block sits.
- Tools used (condition F): `safetensors==0.5.3` (`safe_open`/`save_file`) for
  reading each shard by its index-file mapping and writing the single merged
  output file, and `torch==2.14.0` for the tensor slicing/concatenation. I chose
  plain slicing over `transformers.PreTrainedModel.prune_heads` (also allowed)
  because that API is designed to prune a live model via an internal masking/
  index_select path, not to guarantee a bit-exact, spec-ordered checkpoint edit;
  a slice-and-`torch.cat` on the raw tensors is a direct, zero-computation copy
  of the exact row/column ranges the spec asks for, and is easy to verify against
  the raw shard bit-for-bit (which I did for a spot-checked layer plus the
  untouched MLP tensors before writing this report).
- Approximate time spent, if you can tell: about 10-15 minutes including
  writing and running verification checks against the raw input.
