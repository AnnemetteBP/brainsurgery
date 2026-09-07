# T3 — Participant self-report (condition P)

- **Final artifact path:** `out/T3/solution.py` (output checkpoint in `out/T3/`:
  `model-0000{1..4}-of-00004.safetensors` + `model.safetensors.index.json`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none — the single
  execution passed all checks.

- **Pitfalls or surprises you hit (one line each):**
  - `h.<i>.attn.bias` (the causal-mask buffer to drop) and `h.<i>.attn.c_attn.bias` /
    `h.<i>.attn.c_proj.bias` (projection biases to keep) are one path component
    apart; an unanchored `attn.bias` pattern would have caught neither correctly,
    so I anchored both regexes with `\Z` / `fullmatch` and cross-checked them
    against an explicitly constructed name set of 48 + 12.
  - The safetensors header stores tensors in an arbitrary (offset) order, not
    insertion order, so shard membership depends entirely on the iteration order
    you choose. I iterate `sorted(keys)`, which is also what `safe_open().keys()`
    and hence `load_file()` yield, so the greedy packing is deterministic.
  - `wte.weight` is 154 MB, well over the 64 MiB budget. The standard greedy rule
    ("flush before adding only if the shard is non-empty") already puts it alone
    in its own shard, so no special case was needed — but a naive "reject any
    tensor over the budget" check would have failed the run.
  - Shard *file* sizes are slightly over the budget (66,265,008 vs 66,259,968
    bytes) because of the safetensors header; the budget applies to tensor data
    only, so the check has to sum `numel * element_size`, not `os.path.getsize`.
  - `.to(torch.bfloat16)` is already round-to-nearest-even, so no manual rounding
    was needed; I verified bit-exactness by reloading every shard and comparing
    against `src[name].to(torch.bfloat16)` / `src[name]` with `torch.equal`.

- **Anything in the task text or documentation that was unclear:**
  - The shard file naming convention is not specified. I used the HuggingFace
    convention `model-<i:05d>-of-<n:05d>.safetensors`, which is what the index
    format implies, but a grader comparing filenames literally would depend on
    that guess.
  - The iteration order used for greedy packing is not specified either, and it
    determines exactly which tensor lands in which shard. Alphabetical order is
    the only order that is reproducible from the file, so I assumed it.
  - The index `metadata` block is not mentioned; I included
    `{"total_size": 327889920}` since serving stacks expect it and it is harmless.

- **Tools used (condition F):** n/a — condition P (plain PyTorch + safetensors).
  Versions from the sandbox env: `torch==2.14.0`, `safetensors==0.5.3`,
  `numpy==2.5.2`, CPython 3.13.

- **Approximate time spent, if you can tell:** ~10 minutes. Most of it went to
  reading the safetensors header with shell tools (`od` + `tail` + `jq`) and
  simulating the greedy packing in `awk` first, so that the script's expected
  layout (4 shards; 59 / 58 / 30 / 1 tensors) was known before the one run.

## Result of the single execution

```
model-00001-of-00004.safetensors:  59 tensors, 66,259,968 bytes
model-00002-of-00004.safetensors:  58 tensors, 66,256,896 bytes
model-00003-of-00004.safetensors:  30 tensors, 40,983,552 bytes
model-00004-of-00004.safetensors:   1 tensors, 154,389,504 bytes
OK: 148 tensors, 48 bfloat16, 12 buffers dropped, 4 shards, total_size=327,889,920 bytes
```
