# T3 run report (condition F, GPT-2 124M)

## Participant self-report

- **Final artifact path:** `out/T3/solution.py` (run as
  `.venv/bin/python out/T3/solution.py` from the sandbox root). Output:
  `out/T3/model-0000{1..4}-of-00004.safetensors` + `out/T3/model.safetensors.index.json`.

- **Number of times you executed the script or plan:** 2

- **Which executions failed, and why (one line each):**
  1. Execution 1 produced a correct checkpoint (every check passed) but its
     `shutil.rmtree(out/T3)` "clean the destination" step deleted the authored
     artifacts in that same directory, `solution.py` included. Not a wrong-output
     failure, but the run destroyed its own source, so I count it as failed (`other`).
  2. Execution 2, with the cleanup narrowed to `*.safetensors` / `*.index.json`,
     passed every check and wrote the final output.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious `.*weight` / `endswith("weight")` targeting would have swept in
    `wte.weight`, `wpe.weight`, `ln_1/ln_2/ln_f.weight` — I enumerated the 48
    projection names explicitly (12 layers x 4 suffixes) instead of pattern-matching,
    so overmatch is impossible by construction.
  - `h.<i>.attn.bias` is the `[1,1,1024,1024]` causal-mask buffer, not a bias
    parameter; the real biases are `c_attn.bias`/`c_proj.bias`/`c_fc.bias`. Deleting
    "the attn biases" by name prefix would have destroyed parameters.
  - `wte.weight` is 154 MB, well over the 64 MiB shard budget, so it must land in a
    shard of its own; a naive "pack until full" splitter that refuses to exceed the
    budget would either loop or produce an invalid shard. HF's splitter handles the
    oversized-tensor case, so I checked the budget only for multi-tensor shards.
  - The task requires the script to live under `out/<task>/` — the same directory as
    the output — so the reflex "rmtree the destination before writing" deletes the
    solution. The cleanup has to be scoped to checkpoint files by extension.
  - The obvious condition-F route (`transformers` + `save_pretrained(dtype=...)`) is
    the wrong shape of tool here: its `dtype` is uniform across the whole model, and
    round-tripping through `GPT2Model` would have to reconcile the prefix-less key
    names and the (now removed in transformers 5.x) mask buffer. Not worth the risk
    for a task whose grading is bit-exact.

- **Anything in the task text or documentation that was unclear:**
  - The shard *assignment* is under-specified: the task states the rule (<= 64 MiB of
    tensor data per shard, oversized tensor alone) but not the packing order or the
    shard filename pattern. I assumed the canonical HuggingFace layout
    (`model-0000i-of-0000N.safetensors`, greedy packing in checkpoint key order,
    index with `metadata.total_size` + `weight_map`), which is what serving stacks
    expect and what `save_pretrained` emits. A different but rule-compliant packing
    would produce a different file-to-tensor map.
  - "not counting file headers" is clear, but it means the on-disk file size slightly
    exceeds 64 MiB for full shards (66,265,008 bytes for shard 1); I checked the
    tensor-data total, not the file size.

- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — `load_file` / `save_file`. Direct, dtype-faithful, no model
    class in the way, so tensor names and values pass through untouched.
  - `torch` 2.14.0+cu130 — `tensor.to(torch.bfloat16)` for the required
    round-to-nearest-even cast, and `torch.equal` to prove the untouched tensors are
    bit-identical to the input.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards`, the *same*
    helper `transformers.save_pretrained` uses internally. This gives the canonical
    shard layout and `weight_map` without me reimplementing (and mis-implementing)
    the oversized-tensor rule, while leaving the dtype policy entirely in my hands.
  - Deliberately **not** used: `transformers.save_pretrained(dtype=...)` (uniform
    dtype only, see above), `mergekit` (merge/slice oriented; its `dtype` is also
    global), `peft` (no adapters involved), `torch-state-bridge` (this task renames
    nothing).

- **Approximate time spent, if you can tell:** ~12 minutes, most of it inspecting the
  input key set and deciding the sharding convention.

## Checks enforced by the run

All raise `AssertionError` and abort *before* anything is written:

| Check | Result |
|---|---|
| exactly 48 bfloat16 tensors | 48 |
| `h.0.attn.c_attn.weight` is bfloat16 | yes |
| `wte.weight` is float32 | yes |
| output has exactly 148 tensors | 148 |

Plus supporting checks: all 48 target names exist in the input with the documented
shapes; output key set == input key set minus the 12 mask buffers; no non-projection
tensor is left non-float32; no projection changes shape; every non-cast tensor is
`torch.equal` to its input; every multi-tensor shard is within 67,108,864 bytes of
tensor data; `weight_map` covers every tensor; and after writing, the shards are
re-read from disk and verified for count, dtypes, no cross-shard duplicates, and
bit-exact round-trip.

## Resulting layout

```
model-00001-of-00004.safetensors:  59 tensors,  66,259,968 bytes of tensor data
model-00002-of-00004.safetensors:  58 tensors,  66,256,896 bytes of tensor data
model-00003-of-00004.safetensors:   1 tensor,  154,389,504 bytes  (wte.weight, alone)
model-00004-of-00004.safetensors:  30 tensors,  40,983,552 bytes of tensor data
model.safetensors.index.json:      weight_map over 148 tensors, total_size 327,889,920
```
