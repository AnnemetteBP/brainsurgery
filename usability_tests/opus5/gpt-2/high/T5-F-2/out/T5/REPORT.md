# T5 self-report (condition F, GPT-2 124M)

- **Final artifact path:** `out/T5/solution.py` (run as
  `.venv/bin/python out/T5/solution.py` from the sandbox root).

- **Number of times you executed the script or plan:** 2.

- **Which executions failed, and why (one line each):**
  - Execution 1 produced a correct, fully verified output (all checks passed,
    exit 0), but its cleanup step `shutil.rmtree(out/T5)` deleted the solution
    script itself, since TASK.md requires the authored artifact to live in the
    same directory as the output. No output defect, but the artifact was gone.
  - Execution 2: same script with cleanup narrowed to `*.safetensors` and the
    index file; passed with the identical result.

- **Pitfalls or surprises you hit (one line each):**
  - The base checkpoint carries the 12 `h.<i>.attn.bias` causal-mask buffers
    (4 MiB each), which a `peft.merge_and_unload()` -> `save_pretrained()`
    round-trip would not put back into the output key set; that ruled out the
    "obvious" PEFT route and decided the file-level approach.
  - Conv1D layout: the base `c_attn.weight` is `[in, out] = [768, 2304]` while
    `B @ A` is `[out, in]`, so the product needs `.T` before the add; this is
    exactly what `fan_in_fan_out: true` in `adapter_config.json` records, and I
    read the flag from the config rather than hardcoding the transpose.
  - Adapter names carry the `base_model.model.` prefix and the `.lora_A/.lora_B`
    infix, so the mapping to base names is a strip-prefix + replace-suffix, and
    the two factors of a layer arrive as separate tensors that have to be paired.
  - `adapter_config.json` lists `target_modules: ["c_attn"]` (the bare module
    name), not the `attn.c_attn` spelling used in TASK.md; I keyed off the
    adapter tensor names instead, which is unambiguous.
  - `wte.weight` (154 MB) exceeds the 100 MiB shard budget on its own, so the
    budget check has to exempt a shard that holds exactly one tensor.
  - The authored script and the checkpoint output share `out/T5/`, so a
    `rmtree` of the output directory removes the script; the cleanup has to
    target the checkpoint files by pattern instead (cost me one re-run).
  - `safe_open(...).keys()` returns sorted names while `load_file()` returns
    file-header order; the shard assignment follows the header order, which is
    what `save_pretrained` would also do.

- **Anything in the task text or documentation that was unclear:**
  - The shard budget is stated as a rule (<=100 MiB of tensor data per shard,
    oversized tensor alone) but the grading mentions "sharding rules" compared
    against a hidden reference; it is not stated whether the reference's exact
    tensor-to-shard assignment must match, or only that the rules hold. I used
    the standard HuggingFace greedy splitter in checkpoint order, which is the
    most likely reference behaviour and satisfies the stated rules either way.
  - Shard file naming is not specified; I used the HF convention
    `model-0000k-of-00005.safetensors`.

- **Tools used (condition F): name, version, and why:**
  - `safetensors` 0.5.3 — `load_file` / `save_file` / `safe_open` for reading
    the base and adapter and writing the shards; the task is checkpoint-file
    editing, so this is the direct route.
  - `torch` 2.14.0 — float32 tensor math for `scale * (B @ A).T` and bit-exact
    comparison (`torch.equal`) in the verification pass.
  - `huggingface_hub` 1.16.1 — `split_torch_state_dict_into_shards`, the same
    splitter `transformers.save_pretrained` uses, so the shard layout and the
    `model.safetensors.index.json` format (`metadata.total_size` + `weight_map`)
    match the standard convention instead of being re-implemented by hand.
  - Considered and rejected: `peft` 0.20.0 `merge_and_unload` and
    `transformers` 5.12.1 `save_pretrained`. Correct on the 12 adapted weights,
    but the round-trip goes through a live `GPT2LMHeadModel`, which does not
    round-trip the `attn.bias` mask buffers, and would have needed the output
    key set patched back up afterwards — more work and more risk than the
    file-level script, for no gain. `mergekit` and `torch-state-bridge` do key
    rewriting / task arithmetic, not low-rank folding, so neither fits.

- **Approximate time spent, if you can tell:** roughly 6 minutes: inspect the
  inputs, decide against the PEFT route, write the script with its checks, two
  runs (the first lost to the cleanup bug above), then an independent numeric
  spot-check.

## Checks enforced by the run

Before writing (all raise `CheckFailed` and exit 1):
exactly 12 complete adapter pairs found and every target present in the base;
per-pair rank and shape agreement, base dtype float32, delta shape equal to
base shape; no key containing `lora_` in the output; `h.0.attn.c_attn.weight`
still `[768, 2304]`; exactly 160 output tensors and the key set identical to
the base.

After writing, the script re-reads the shards from disk and re-asserts: the
`weight_map` covers exactly the 160 tensors, referenced shard files equal the
files on disk, no tensor is duplicated across shards, every shard's tensor
payload is within 100 MiB unless it holds a single tensor, dtypes and shapes
survive the round-trip, the 12 merged weights are present, and the 148
untouched tensors are bit-exact against `inputs/base/model.safetensors`.

Independent spot-check (float64, outside the script): worst relative Frobenius
error across the 12 merged weights vs `base + 2 * (B @ A).T` is 2.6e-8, against
the 1e-5 tolerance. Output: 5 shards, payloads 97.1 / 99.9 / 95.4 / 147.2
(`wte.weight` alone) / 83.1 MiB, 160 tensors total.
