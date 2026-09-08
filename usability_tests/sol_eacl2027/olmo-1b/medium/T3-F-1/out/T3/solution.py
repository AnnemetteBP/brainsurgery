#!/usr/bin/env python3
"""Create the mixed-precision, sharded T3 checkpoint."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
TARGET_KEYS = {
    f"model.layers.{layer}.{projection}.weight"
    for layer in range(16)
    for projection in PROJECTIONS
}


def load_index():
    with (BASE / INDEX_NAME).open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index["weight_map"]
    assert len(weight_map) == 114, f"expected 114 input tensors, got {len(weight_map)}"
    assert len(TARGET_KEYS) == 112
    missing = TARGET_KEYS - weight_map.keys()
    assert not missing, f"missing projection tensors: {sorted(missing)}"
    return weight_map


def read_tensor(weight_map, key):
    with safe_open(BASE / weight_map[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def output_dtype(key):
    return torch.bfloat16 if key in TARGET_KEYS else torch.float32


def validate_before_writing(weight_map):
    """Perform all task-mandated checks before creating checkpoint files."""
    output_keys = set(weight_map)
    assert len(output_keys) == 114, f"output plan has {len(output_keys)} tensors, not 114"

    bf16_count = sum(output_dtype(key) == torch.bfloat16 for key in output_keys)
    assert bf16_count == 112, f"output plan has {bf16_count} bfloat16 tensors, not 112"
    assert output_dtype("model.layers.0.self_attn.q_proj.weight") == torch.bfloat16
    assert output_dtype("model.embed_tokens.weight") == torch.float32

    # Check source types and exercise the exact conversion before any file is written.
    for key in sorted(output_keys):
        tensor = read_tensor(weight_map, key)
        assert tensor.dtype == torch.float32, f"unexpected source dtype for {key}: {tensor.dtype}"
        converted = tensor.to(output_dtype(key))
        assert converted.dtype == output_dtype(key), f"failed dtype conversion for {key}"


def tensor_bytes(weight_map, key):
    tensor = read_tensor(weight_map, key)
    element_size = 2 if key in TARGET_KEYS else 4
    return tensor.numel() * element_size


def plan_shards(weight_map):
    """Greedily shard sorted keys, placing each oversize tensor alone."""
    shards = []
    current = []
    current_bytes = 0
    for key in sorted(weight_map):
        size = tensor_bytes(weight_map, key)
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([key])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(key)
            current_bytes += size
    if current:
        shards.append(current)
    return shards


def main():
    weight_map = load_index()
    validate_before_writing(weight_map)
    shard_plan = plan_shards(weight_map)
    shard_count = len(shard_plan)
    output_weight_map = {}
    total_size = 0

    for number, keys in enumerate(shard_plan, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        tensors = {}
        shard_size = 0
        for key in keys:
            tensor = read_tensor(weight_map, key).to(output_dtype(key)).contiguous()
            tensors[key] = tensor
            size = tensor.numel() * tensor.element_size()
            shard_size += size
            total_size += size
            output_weight_map[key] = filename
        assert shard_size <= MAX_SHARD_BYTES or len(keys) == 1, (
            f"invalid shard {filename}: {shard_size} bytes across {len(keys)} tensors"
        )
        save_file(tensors, OUTPUT / filename)

    assert len(output_weight_map) == 114
    assert set(output_weight_map) == set(weight_map)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    with (OUTPUT / INDEX_NAME).open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote {len(output_weight_map)} tensors in {shard_count} shards ({total_size} bytes)")


if __name__ == "__main__":
    main()
