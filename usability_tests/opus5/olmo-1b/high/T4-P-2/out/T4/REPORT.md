# T4 run record — Participant self-report

- **Final artifact path:** `out/T4/solution.py` (output: `out/T4/model.safetensors`)

- **Number of times you executed the script or plan:** 1

- **Which executions failed, and why (one line each):**
  - None; the single execution succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - The base is a *sharded* directory while ft1/ft2 are single files, so the loader
    had to unify both layouts; I also cross-checked `model.safetensors.index.json`
    against the keys actually present in the two shards instead of trusting either alone.
  - The stated arithmetic hazard: both task vectors must be taken against the
    unmodified base, so I computed `base + λ*(ft1-base) + λ*(ft2-base)` in one
    expression from freshly read base tensors rather than accumulating in place.
  - Ordering requirement: the full 66-tensor shared verification runs to completion
    before any merge is computed, so I cached the verified base tensors during that
    pass and built the output from them (also avoids re-reading 5 GB).
  - `==`/`torch.equal` on floats is not a bit-exact test (NaN != NaN, `-0.0 == 0.0`),
    and grading is bit-exact for the 66 unchanged tensors, so I compared the raw byte
    views (`.view(torch.uint8)`) instead.
  - `lm_head.weight` and `model.embed_tokens.weight` are separate entries in this
    checkpoint; reading each with `get_tensor` gives independent storages, so
    `save_file` did not trip over shared memory.
  - Memory: three 5 GB float32 checkpoints cannot be held at once, so tensors are
    read lazily one name at a time via `safe_open`, with only the output dict resident.

- **Anything in the task text or documentation that was unclear:**
  - "identical" for the 66 shared tensors is not qualified; I read it as bit-exact
    (no tolerance), which is consistent with the bit-exact grading of those tensors.
  - The output directory is described as a single file `out/T4/model.safetensors`
    with no mention of `config.json`/tokenizer files, so I wrote only that file.
  - The MLP set is given both as a name pattern and as a count of 48; I asserted both
    (regex match set == the explicit 16x3 name set == 48) in case they disagreed.

- **Tools used (condition F):** n/a — condition P (torch 2.14.0, safetensors 0.5.3, numpy 2.5.2).

- **Approximate time spent, if you can tell:** ~5 minutes; the script itself runs in ~23 s.
