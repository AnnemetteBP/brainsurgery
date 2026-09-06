# T3 — Participant self-report (condition F, OLMo-1B-0724-hf)

- **Final artifact path:** `out/T3/solution.py` (entry point `out/T3/run.sh`);
  output checkpoint in `out/T3/` (10 shards + `model.safetensors.index.json`).

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):** none; the single
  execution succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - The obvious `.*weight` regex would hit `model.embed_tokens.weight` and
    `lm_head.weight` too, so I enumerated the 112 projection names from the
    16 layers × 7 leaf names instead of pattern-matching, which makes
    overreach impossible rather than merely checked-for.
  - OLMo-1B-0724 has non-parametric layer norms and no attention biases, so
    "everything else" is only the two embedding matrices — 112 + 2 = 114
    accounts for the whole checkpoint, and the "drop the buffers" step is a
    no-op here, as the task states.
  - Shard arithmetic lands exactly on the boundary: one layer is 128 MiB in
    bfloat16 (4×2048² + 3×8192×2048, 2 bytes each), so two layers total
    exactly 268,435,456 bytes. "At most 256 MiB" admits that, and HF's
    splitter uses a strict `>` comparison, so it packs 2 layers per shard
    rather than 1.5. Getting this wrong by one byte of slack would have
    halved shard occupancy.
  - `safe_open(...).keys()` returns keys sorted, not in physical file order,
    so I took the key order from the input `weight_map` instead. Shard
    membership turns out to be insensitive to this: alphabetical layer order
    still pairs (0,1), (10,11), (12,13), (14,15), (2,3), (4,5), (6,7), (8,9)
    — every pair consecutive — so a reference built in numeric `state_dict()`
    order gets the same groupings, only different shard file numbering.
  - I ran the required checks *before* creating `out/T3/`, so a failed check
    cannot leave a partial output directory behind.

- **Anything in the task text or documentation that was unclear:**
  - The 256 MiB budget is given as "at most", but whether the reference used
    `>` or `>=` against it is exactly the difference between 8 layer-shards
    and 16. I went with `>` (2 layers per shard) since "at most" admits
    equality; a stated tie-break would have removed the guess.
  - The spec fixes shard *contents* but not shard *file numbering*, and the
    numbering depends on the key order the splitter is fed. I assumed grading
    checks membership and the index's self-consistency rather than exact
    filenames.
  - Requirement 3 ("do not delete anything") and the objective's "drop
    non-parameter buffers" read as contradictory until the Input section
    clarifies this checkpoint has no buffers.

- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — `safe_open` to read tensors one at a time (casting
    each on the way in keeps peak memory near the ~2.8 GB output rather than
    the 5.1 GB input) and `save_file` to write shards.
  - `torch` 2.14.0 — `tensor.to(torch.bfloat16)`, which is the exact
    round-to-nearest-even cast the task specifies.
  - `huggingface_hub` (pinned) — `split_torch_state_dict_into_shards`, the
    same splitter `save_pretrained` uses. I wanted the shard budget and the
    "a tensor larger than the budget gets its own shard" rule to come from
    the library rather than from my own loop, since that rule is precisely
    what serving stacks expect and is easy to hand-roll subtly wrong.
  - **Deliberately not used:** `transformers` `save_pretrained` with a dtype,
    which `F-allowed.md` suggests as the route for T3. Its `dtype=`/
    `torch_dtype=` argument is uniform across the checkpoint, so it cannot
    express "bfloat16 projections, float32 embeddings" without mutating
    individual parameters on a loaded model first — at which point it adds a
    full `OlmoForCausalLM` instantiation and its own dtype/tying behaviour
    between me and the bytes, for no benefit over calling the splitter
    directly. `mergekit` and `peft` are not applicable to a dtype export.

- **Approximate time spent, if you can tell:** roughly 10 minutes, most of it
  reading the input layout and working out the shard-boundary arithmetic; the
  export itself runs in a few seconds.

## Verification performed

The script's own checks (all run before any file is written; each raises and
exits non-zero):

- input index lists 114 tensors and every input tensor is float32;
- the 112 enumerated projection names are all present in the input;
- exactly 112 output tensors are bfloat16, **and** that set equals the
  intended projection set (catches both under- and over-matching);
- `model.layers.0.self_attn.q_proj.weight` is bfloat16;
- `model.embed_tokens.weight` and `lm_head.weight` are float32;
- every non-projection tensor is float32;
- all 112 cast matrices have the shapes the task names;
- the output has exactly 114 tensors;
- the `weight_map` covers every tensor, and any shard over the 256 MiB budget
  holds exactly one tensor.

Separately, I re-read the written directory and confirmed against the input:
10 shard files; `weight_map` agrees with the file each tensor is actually in;
114 tensors on disk; dtype census 112 bfloat16 / 2 float32; the two 412 MB
embedding matrices alone in their own shards and the other 8 shards at exactly
268,435,456 bytes; unchanged key set and shapes; and bit-exact values compared
as raw integers — the 112 cast tensors match `src.to(torch.bfloat16)` bit for
bit, and the 2 float32 tensors are bitwise identical to the input.
