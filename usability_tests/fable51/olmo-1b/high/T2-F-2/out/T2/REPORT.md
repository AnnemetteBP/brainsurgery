# T2 report (condition F, OLMo-1B-0724-hf)

## Participant self-report

- Final artifact path: `out/T2/solution.py` (produces `out/T2/model.safetensors`).
  Auxiliary, not needed to produce the output: `out/T2/verify.py` (bit-exact
  comparison against the input slices), `out/T2/load_check.py` (loads the
  result into Transformers with 15 heads and runs a forward pass).
- Number of times you executed the script or plan: 1
- Which executions failed, and why (one line each): none; the first run succeeded.
- Pitfalls or surprises you hit (one line each):
  - None in the task itself. The layout was fully specified (Linear `[out, in]`,
    heads as row blocks on q/k/v, column blocks on o_proj, head 5 = index 640..767).
  - For the load check the OLMo config needs `head_dim=128` set explicitly along
    with `num_attention_heads=15` and `num_key_value_heads=15`, otherwise
    hidden_size / heads would not be an integer.
- Anything in the task text or documentation that was unclear: nothing.
- Tools used (condition F): name, version, and why:
  - `safetensors` 0.5.3: read the two input shards via the index file, write the
    single output file. Chosen because the grade is bit-exact on values and key
    set; direct tensor slicing avoids any library-side renaming or re-layout.
  - `torch` 2.14.0: `index_select` on the row/column axis with an explicit keep
    index (0..639, 768..2047), `.contiguous()` before saving.
  - `transformers` 5.12.1: only for the post-hoc load check (`from_config` with
    15 heads, `load_state_dict` strict key match, forward pass gives finite logits
    and predicts " Paris"). I did not use `prune_heads` because it operates on a
    loaded model and would also require config rewriting and `save_pretrained`,
    which adds sharding and metadata decisions unrelated to the required output
    (one file, 114 tensors, exact values).
- Approximate time spent, if you can tell: about 5 minutes.

## Checks enforced by `solution.py` before writing

- input has exactly 114 tensors, no duplicate keys across shards;
- each q/k/v/o projection has shape `[2048, 2048]` before slicing;
- `model.layers.0.self_attn.{q,k,v}_proj.weight` is `[1920, 2048]` and
  `o_proj.weight` is `[2048, 1920]` (required checks), plus the same for all 16 layers;
- every tensor is float32; the output still has exactly 114 tensors;
- the destination file does not already exist.

`verify.py` confirmed independently that all 64 pruned tensors equal
`cat(a[:640], a[768:])` (rows) or `cat(a[:, :640], a[:, 768:])` (columns) of the
input and the remaining 50 tensors are unchanged.
