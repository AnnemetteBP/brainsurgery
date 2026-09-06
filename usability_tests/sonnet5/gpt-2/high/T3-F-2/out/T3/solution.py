#!/usr/bin/env python
"""T3: mixed-precision export with sharding, for GPT-2 (124M).

Plain script on top of `torch` + `safetensors` (both in F-allowed.md). No
merging/adapter framework is actually needed here: the task is a per-tensor
dtype cast plus a byte-budgeted repack, which the two base libraries do
directly and let us assert on before anything is written.

Steps:
  1. Load every tensor from inputs/base/model.safetensors (all float32).
  2. Cast exactly the 48 projection matrices (attn.c_attn/c_proj,
     mlp.c_fc/c_proj weights, all 12 layers) to bfloat16 with `.to()`
     (round-to-nearest-even). Leave every other tensor's values untouched.
  3. Drop the 12 `h.<i>.attn.bias` causal-mask buffers (not parameters).
  4. Run the required checks; abort loudly if any fails.
  5. Greedily bin-pack tensors (in a fixed, deterministic model order) into
     shards of at most 64 MiB (67,108,864 bytes) of tensor data each; a
     tensor that alone exceeds the budget (wte.weight) gets its own shard.
  6. Write shard files + model.safetensors.index.json (weight_map + total
     tensor byte size), HF sharded-checkpoint layout.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SANDBOX_ROOT = HERE.parent.parent
INPUT_PATH = SANDBOX_ROOT / "inputs" / "base" / "model.safetensors"
OUT_DIR = SANDBOX_ROOT / "out" / "T3"

NUM_LAYERS = 12
SHARD_BUDGET_BYTES = 64 * 1024 * 1024  # 67,108,864 bytes, tensor data only

BF16_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def bf16_targets() -> set[str]:
    return {f"h.{i}.{suf}" for i in range(NUM_LAYERS) for suf in BF16_SUFFIXES}


def buffers_to_drop() -> set[str]:
    return {f"h.{i}.attn.bias" for i in range(NUM_LAYERS)}


def model_key_order() -> list[str]:
    """Fixed, deterministic traversal order (numeric layer order, not
    lexicographic — h.10/h.11 must not land between h.1 and h.2)."""
    order = ["wte.weight", "wpe.weight"]
    for i in range(NUM_LAYERS):
        order += [
            f"h.{i}.attn.bias",
            f"h.{i}.attn.c_attn.weight",
            f"h.{i}.attn.c_attn.bias",
            f"h.{i}.attn.c_proj.weight",
            f"h.{i}.attn.c_proj.bias",
            f"h.{i}.ln_1.weight",
            f"h.{i}.ln_1.bias",
            f"h.{i}.ln_2.weight",
            f"h.{i}.ln_2.bias",
            f"h.{i}.mlp.c_fc.weight",
            f"h.{i}.mlp.c_fc.bias",
            f"h.{i}.mlp.c_proj.weight",
            f"h.{i}.mlp.c_proj.bias",
        ]
    order += ["ln_f.weight", "ln_f.bias"]
    return order


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def pack_shards(keys_in_order: list[str], sizes: dict[str, int]) -> list[list[str]]:
    """Greedy bin packing in a fixed order: fill a shard until the next
    tensor would overflow the budget, then start a new one. A tensor whose
    own size exceeds the budget is placed alone in its own shard."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in keys_in_order:
        size = sizes[key]
        if size > SHARD_BUDGET_BYTES:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([key])
            continue
        if current and current_size + size > SHARD_BUDGET_BYTES:
            shards.append(current)
            current, current_size = [], 0
        current.append(key)
        current_size += size
    if current:
        shards.append(current)
    return shards


def main() -> None:
    if not INPUT_PATH.exists():
        sys.exit(f"input checkpoint not found: {INPUT_PATH}")

    state_dict = load_file(str(INPUT_PATH))  # framework="pt" default, preserves dtype
    input_keys = set(state_dict.keys())

    bf16_set = bf16_targets()
    drop_set = buffers_to_drop()

    missing_bf16 = bf16_set - input_keys
    missing_drop = drop_set - input_keys
    if missing_bf16 or missing_drop:
        sys.exit(f"expected keys not found in input: {missing_bf16 | missing_drop}")

    output: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key in drop_set:
            continue
        if key in bf16_set:
            tensor = tensor.to(torch.bfloat16)
        output[key] = tensor.contiguous()

    # --- Required checks: fail loudly before writing anything ---
    bf16_count = sum(1 for t in output.values() if t.dtype == torch.bfloat16)
    assert bf16_count == 48, f"expected exactly 48 bfloat16 tensors, got {bf16_count}"
    assert output["h.0.attn.c_attn.weight"].dtype == torch.bfloat16, (
        "h.0.attn.c_attn.weight must be bfloat16"
    )
    assert output["wte.weight"].dtype == torch.float32, "wte.weight must be float32"
    assert len(output) == 148, f"expected exactly 148 output tensors, got {len(output)}"
    # Every tensor is either in the bf16 set (bfloat16) or must be float32,
    # unchanged from the input in dtype and values.
    for key, tensor in output.items():
        if key in bf16_set:
            continue
        assert tensor.dtype == torch.float32, f"{key} should stay float32, got {tensor.dtype}"
        assert torch.equal(tensor, state_dict[key]), f"{key} values changed unexpectedly"
    assert not (drop_set & output.keys()), "a dropped buffer leaked into the output"
    assert set(output.keys()) == (input_keys - drop_set), "tensor name set does not match"

    # --- Shard ---
    order = model_key_order()
    assert set(order) == input_keys, "fixed key order does not cover the input checkpoint"
    ordered_output_keys = [k for k in order if k in output]

    sizes = {k: tensor_nbytes(v) for k, v in output.items()}
    shards = pack_shards(ordered_output_keys, sizes)

    for shard_keys in shards:
        total = sum(sizes[k] for k in shard_keys)
        assert total <= SHARD_BUDGET_BYTES or len(shard_keys) == 1, (
            "shard exceeds the byte budget without being a lone oversized tensor"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for idx, shard_keys in enumerate(shards, start=1):
        shard_name = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        shard_tensors = {k: output[k] for k in shard_keys}
        save_file(shard_tensors, str(OUT_DIR / shard_name), metadata={"format": "pt"})
        for k in shard_keys:
            weight_map[k] = shard_name
            total_size += sizes[k]

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(OUT_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {len(output)} tensors across {n_shards} shard(s) to {OUT_DIR}")
    print(f"total tensor bytes: {total_size}")


if __name__ == "__main__":
    main()
