#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a sharded safetensors checkpoint."""

import json
from math import prod
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUTPUT_DIR = ROOT / "out" / "T5"
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tensor_metadata(weight_map):
    """Return tensor byte sizes and shapes without loading tensor payloads."""
    sizes = {}
    shapes = {}
    handles = {}
    try:
        for name, shard_name in weight_map.items():
            if shard_name not in handles:
                handles[shard_name] = safe_open(
                    BASE_DIR / shard_name, framework="pt", device="cpu"
                )
            view = handles[shard_name].get_slice(name)
            shape = tuple(view.get_shape())
            dtype = view.get_dtype()
            if dtype not in DTYPE_BYTES:
                raise AssertionError(f"Unsupported safetensors dtype {dtype!r} for {name}")
            shapes[name] = shape
            sizes[name] = prod(shape) * DTYPE_BYTES[dtype]
    finally:
        handles.clear()
    return sizes, shapes


def adapter_pairs(adapter_path: Path):
    """Map each base weight name to its LoRA A and B tensor names."""
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        adapter_names = set(adapter.keys())

    pairs = {}
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    prefix = "base_model.model."
    for adapter_name in adapter_names:
        if not adapter_name.endswith(a_suffix):
            continue
        if not adapter_name.startswith(prefix):
            raise AssertionError(f"Unexpected adapter prefix: {adapter_name}")
        b_name = adapter_name[: -len(a_suffix)] + b_suffix
        if b_name not in adapter_names:
            raise AssertionError(f"Missing B factor for {adapter_name}")
        base_name = adapter_name[len(prefix) : -len(a_suffix)] + ".weight"
        if base_name in pairs:
            raise AssertionError(f"Duplicate adapter pair for {base_name}")
        pairs[base_name] = (adapter_name, b_name)

    paired_adapter_names = {name for pair in pairs.values() for name in pair}
    if paired_adapter_names != adapter_names:
        extras = sorted(adapter_names - paired_adapter_names)
        raise AssertionError(f"Unpaired or unexpected adapter tensors: {extras}")
    return pairs


def plan_shards(names, sizes):
    """Greedily pack tensors in index order, respecting tensor-data limit."""
    shards = []
    current = []
    current_bytes = 0
    for name in names:
        size = sizes[name]
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
        # An individually oversized tensor must be its own shard.
        if size > MAX_SHARD_BYTES:
            if len(current) != 1:
                raise AssertionError("Oversized tensor was not isolated")
            shards.append(current)
            current = []
            current_bytes = 0
        elif current_bytes > MAX_SHARD_BYTES:
            raise AssertionError("Shard plan exceeds the byte limit")
    if current:
        shards.append(current)
    return shards


def main():
    base_index = read_json(BASE_DIR / INDEX_NAME)
    config = read_json(LORA_DIR / "adapter_config.json")
    weight_map = base_index["weight_map"]
    output_names = list(weight_map)
    adapter_path = LORA_DIR / "adapter_model.safetensors"
    pairs = adapter_pairs(adapter_path)
    sizes, shapes = tensor_metadata(weight_map)

    if config.get("fan_in_fan_out") is not False:
        raise AssertionError("This script expects fan_in_fan_out=false")
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if not isinstance(rank, (int, float)) or rank == 0:
        raise AssertionError(f"Invalid LoRA rank: {rank!r}")
    scale = float(alpha) / float(rank)

    # Required checks, all deliberately completed before any checkpoint write.
    if len(pairs) != 32:
        raise AssertionError(f"Expected exactly 32 adapter pairs, found {len(pairs)}")
    missing_base = sorted(set(pairs) - set(output_names))
    if missing_base:
        raise AssertionError(f"Adapter targets absent from base: {missing_base}")
    if any("lora_" in name for name in output_names):
        raise AssertionError("Output tensor names contain a lora_ tensor")
    probe = "model.layers.0.self_attn.q_proj.weight"
    if shapes.get(probe) != (2048, 2048):
        raise AssertionError(f"{probe} has unexpected shape {shapes.get(probe)}")
    if len(output_names) != 114:
        raise AssertionError(f"Expected exactly 114 output tensors, found {len(output_names)}")

    shards = plan_shards(output_names, sizes)
    total_shards = len(shards)
    output_weight_map = {}
    total_size = sum(sizes.values())

    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        for shard_number, tensor_names in enumerate(shards, start=1):
            output_shard_name = (
                f"model-{shard_number:05d}-of-{total_shards:05d}.safetensors"
            )
            tensors = {}
            base_handles = {}
            try:
                for name in tensor_names:
                    source_shard = weight_map[name]
                    if source_shard not in base_handles:
                        base_handles[source_shard] = safe_open(
                            BASE_DIR / source_shard, framework="pt", device="cpu"
                        )
                    tensor = base_handles[source_shard].get_tensor(name)
                    if name in pairs:
                        a_name, b_name = pairs[name]
                        a = adapter.get_tensor(a_name)
                        b = adapter.get_tensor(b_name)
                        if tensor.dtype != torch.float32 or a.dtype != torch.float32 or b.dtype != torch.float32:
                            raise AssertionError(f"Non-float32 merge inputs for {name}")
                        if tuple(b.shape) != (tensor.shape[0], rank) or tuple(a.shape) != (rank, tensor.shape[1]):
                            raise AssertionError(
                                f"LoRA/base shape mismatch for {name}: "
                                f"base={tuple(tensor.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
                            )
                        tensor = tensor + (b @ a) * scale
                    tensors[name] = tensor.contiguous()
                    output_weight_map[name] = output_shard_name

                shard_bytes = sum(
                    value.numel() * value.element_size() for value in tensors.values()
                )
                if shard_bytes > MAX_SHARD_BYTES and len(tensors) != 1:
                    raise AssertionError(
                        f"{output_shard_name} exceeds 512 MiB with multiple tensors"
                    )
                save_file(tensors, OUTPUT_DIR / output_shard_name)
            finally:
                base_handles.clear()

    if set(output_weight_map) != set(output_names):
        raise AssertionError("Internal error: incomplete output weight map")
    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    with (OUTPUT_DIR / INDEX_NAME).open("w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Merged {len(pairs)} LoRA pairs and wrote {len(output_names)} tensors "
        f"to {total_shards} shards ({total_size} tensor bytes)."
    )


if __name__ == "__main__":
    main()
