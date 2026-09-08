#!/usr/bin/env python3
"""Merge a PEFT LoRA into GPT-2 tensors and write a sharded checkpoint."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUTPUT_DIR = Path(__file__).resolve().parent
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 100 * 1024 * 1024


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def pack_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Greedily pack in key order, leaving over-limit tensors alone."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name, tensor in tensors.items():
        size = tensor_bytes(tensor)
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = {}
                current_bytes = 0
            shards.append({name: tensor})
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current = {}
                current_bytes = 0
            current[name] = tensor
            current_bytes += size

    if current:
        shards.append(current)
    return shards


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)

    rank = config["r"]
    alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    if fan_in_fan_out is not True:
        raise ValueError("This GPT-2 Conv1D merge requires fan_in_fan_out=true")
    scale = float(alpha) / float(rank)

    base = load_file(BASE_PATH, device="cpu")
    adapter = load_file(ADAPTER_PATH, device="cpu")
    output = dict(base)

    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    prefix = "base_model.model."
    a_names = sorted(name for name in adapter if name.endswith(a_suffix))
    b_names = {name for name in adapter if name.endswith(b_suffix)}

    if len(a_names) != 12 or len(b_names) != 12:
        raise ValueError(
            f"Expected exactly 12 LoRA A/B tensors, found {len(a_names)} A and {len(b_names)} B"
        )

    merged_count = 0
    used_adapter_names: set[str] = set()
    for a_name in a_names:
        stem = a_name[: -len(a_suffix)]
        b_name = stem + b_suffix
        if b_name not in adapter:
            raise KeyError(f"Missing matching B tensor for {a_name}")
        if not stem.startswith(prefix):
            raise ValueError(f"Unexpected adapter namespace: {a_name}")
        base_name = stem[len(prefix) :] + ".weight"
        if base_name not in output:
            raise KeyError(f"Adapter destination is absent from base: {base_name}")

        a = adapter[a_name]
        b = adapter[b_name]
        weight = output[base_name]
        if a.dtype != torch.float32 or b.dtype != torch.float32 or weight.dtype != torch.float32:
            raise TypeError(f"Expected float32 tensors for {base_name}")
        if a.shape != (rank, weight.shape[0]) or b.shape != (weight.shape[1], rank):
            raise ValueError(
                f"Incompatible shapes for {base_name}: weight={tuple(weight.shape)}, "
                f"A={tuple(a.shape)}, B={tuple(b.shape)}"
            )

        update = (b @ a).T * scale
        if update.shape != weight.shape:
            raise ValueError(f"LoRA update shape mismatch for {base_name}")
        output[base_name] = (weight + update).contiguous()
        used_adapter_names.update((a_name, b_name))
        merged_count += 1

    if used_adapter_names != set(adapter):
        unused = sorted(set(adapter) - used_adapter_names)
        raise ValueError(f"Unrecognized or unused adapter tensors: {unused}")

    # Required pre-write checks.
    if merged_count != 12:
        raise ValueError(f"Expected exactly 12 merged adapter pairs, got {merged_count}")
    if any("lora_" in name for name in output):
        raise ValueError("Output contains an adapter tensor name")
    if output.get("h.0.attn.c_attn.weight", torch.empty(0)).shape != (768, 2304):
        raise ValueError("h.0.attn.c_attn.weight does not have shape [768, 2304]")
    if len(output) != 160:
        raise ValueError(f"Expected exactly 160 output tensors, got {len(output)}")
    if set(output) != set(base):
        raise ValueError("Output key set differs from the base checkpoint")

    shards = pack_shards(output)
    for shard in shards:
        size = sum(tensor_bytes(tensor) for tensor in shard.values())
        if size > MAX_SHARD_BYTES and len(shard) != 1:
            raise ValueError(f"Multi-tensor shard exceeds limit: {size} bytes")

    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    filenames: list[str] = []
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        filenames.append(filename)
        for name in shard:
            if name in weight_map:
                raise ValueError(f"Tensor assigned more than once: {name}")
            weight_map[name] = filename

    if set(weight_map) != set(output):
        raise ValueError("Shard index does not cover the output exactly")

    for filename, shard in zip(filenames, shards, strict=True):
        save_file(shard, OUTPUT_DIR / filename)

    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in output.values())},
        "weight_map": weight_map,
    }
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Merged {merged_count} LoRA pairs into {len(output)} tensors across {shard_count} shards")


if __name__ == "__main__":
    main()
