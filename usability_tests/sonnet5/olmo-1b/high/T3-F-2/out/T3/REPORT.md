# Participant self-report — T3-F-2 (OLMo-1B-0724-hf)

- **Final artifact path:** `out/T3/solution.py` (invoked via `out/T3/run.sh`)
- **Number of times you executed the script or plan:** 1
- **Which executions failed, and why (one line each):** none; the single run succeeded.
- **Pitfalls or surprises you hit (one line each):**
  - None of the allowed higher-level tools (`mergekit`, `torch-state-bridge`,
    `transformers` sharded export) give direct per-tensor "cast this regex-matched
    subset to bf16, leave everything else float32 bit-exact, and repack shards
    against a byte budget" in one step, so a plain script on `safetensors` +
    `torch` was the most direct route rather than a workaround.
  - `save_file`'s on-disk file size includes a header, so the shard-size check
    has to sum raw tensor bytes (`numel * element_size`) rather than compare
    against `os.path.getsize` of the written file.
- **Anything in the task text or documentation that was unclear:** No — the
  exact tensor names, shapes, and the 256 MiB budget (with the oversized-tensor
  exception for the two 412 MB tensors) were fully specified.
- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — load every input shard tensor-by-tensor and write
    the output shards; it's the format both input and output use.
  - `torch` 2.14.0 — `.to(torch.bfloat16)` for the round-to-nearest-even cast
    on exactly the 112 named projection matrices, and `.contiguous()` before
    saving.
  - Plain Python (`re`, `json`) for the projection-name regex, the greedy
    shard bin-packer, and the required-checks assertions (each fails loudly
    with `AssertionError` before any file is written). No merge/adapter tool
    (`mergekit`, `peft`, `torch-state-bridge`, `transformers` export) was
    needed since this task is pure per-tensor dtype casting plus resharding,
    not architecture editing, head pruning, or adapter merging.
- **Approximate time spent, if you can tell:** ~10 minutes.

## Verification performed

- Required checks (enforced in `solution.py::run_required_checks`, run before
  any output is written): exactly 112 bfloat16 tensors, all of which match the
  projection-matrix name pattern; `model.layers.0.self_attn.q_proj.weight` is
  bfloat16; `model.embed_tokens.weight` and `lm_head.weight` are float32;
  total tensor count is 114.
- Post-hoc spot checks (not part of the script, run manually after):
  `model.embed_tokens.weight` is bit-exact (`torch.equal`) against the input;
  `model.layers.0.self_attn.q_proj.weight` in the output matches
  `orig.to(torch.bfloat16)` exactly.
- All 10 output shards are within the 256 MiB tensor-data budget; the two
  412 MB tensors (`model.embed_tokens.weight`, `lm_head.weight`) each sit
  alone in their own shard.
