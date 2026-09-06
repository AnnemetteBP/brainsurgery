# Participant self-report — T3, condition F

## Tools used

- `safetensors` (`safe_open` / `save_file`) for reading and writing the checkpoint.
- `torch` for the dtype casts (`.to(torch.bfloat16)`, `.to(torch.float32)`) and tensor byte-size math.

I did not use `transformers`, `mergekit`, `peft`, or `torch-state-bridge`: the
task is a direct dtype-cast-and-reshard on a raw safetensors file with no
architecture reshaping, adapter merging, or key renaming involved, so a plain
script on `safetensors`+`torch` is the smallest correct tool for it and avoids
routing the checkpoint through a full HF model object.

## Approach

1. Iterate all keys in `inputs/base/model.safetensors`; skip the 12
   `h.<i>.attn.bias` buffers (regex `^h\.\d+\.attn\.bias$`).
2. Cast the 48 projection-weight tensors (regex
   `^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$`) to
   bfloat16; keep everything else float32.
3. Assert all four required checks (48 bf16 tensors, `h.0.attn.c_attn.weight`
   is bf16, `wte.weight` is fp32, exactly 148 output tensors) plus two extra
   sanity assertions (no buffer leaked through, every tensor's dtype matches
   its intended bucket) before any file is written.
4. Greedily bin-pack tensors into shards of at most 64 MiB of tensor data
   (in input-file key order); a tensor larger than the limit (`wte.weight`)
   is placed alone in its own shard.
5. Write each shard with `safetensors.torch.save_file` and build
   `model.safetensors.index.json` with `weight_map` and `metadata.total_size`.

## Result

Ran `python solution.py` (also wrapped in `run.sh`). Single execution,
succeeded on the first attempt — 148 tensors written across 4 shards
(3 shards ≤ 64 MiB, `wte.weight` alone in the 4th at ~154 MB), 48 bfloat16 /
100 float32, all required-check assertions passed.

## Executions

- executions: 1
- failed_executions: 0
- first_execution_success: yes
