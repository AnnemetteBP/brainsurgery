#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA adapter into GPT-2 and export shards."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUTPUT_DIR = ROOT / "out/T5"
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 100 * 1024 * 1024


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def pack_shards(
    tensors: dict[str, torch.Tensor], max_bytes: int
) -> list[list[str]]:
    """Greedily pack keys, storing any individually oversized tensor alone."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in sorted(tensors):
        size = tensor_nbytes(tensors[name])
        if size > max_bytes:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
        elif current and current_bytes + size > max_bytes:
            shards.append(current)
            current = [name]
            current_bytes = size
        else:
            current.append(name)
            current_bytes += size

    if current:
        shards.append(current)
    return shards


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)

    rank = config["r"]
    alpha = config["lora_alpha"]
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"invalid LoRA rank: {rank!r}")
    if config.get("fan_in_fan_out") is not True:
        raise ValueError("this GPT-2 merge requires fan_in_fan_out=true")
    scale = float(alpha) / rank

    base = load_file(BASE_PATH, device="cpu")
    adapter = load_file(ADAPTER_PATH, device="cpu")

    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    adapter_prefix = "base_model.model."
    a_keys = {key[: -len(a_suffix)]: key for key in adapter if key.endswith(a_suffix)}
    b_keys = {key[: -len(b_suffix)]: key for key in adapter if key.endswith(b_suffix)}
    if set(a_keys) != set(b_keys):
        missing_a = sorted(set(b_keys) - set(a_keys))
        missing_b = sorted(set(a_keys) - set(b_keys))
        raise RuntimeError(
            f"unpaired adapter tensors: missing A={missing_a}, missing B={missing_b}"
        )
    if len(a_keys) != 12:
        raise RuntimeError(f"expected exactly 12 adapter pairs, found {len(a_keys)}")
    if len(adapter) != 2 * len(a_keys):
        extra = sorted(
            key
            for key in adapter
            if not key.endswith(a_suffix) and not key.endswith(b_suffix)
        )
        raise RuntimeError(f"unexpected non-paired adapter tensors: {extra}")

    merged_count = 0
    for stem in sorted(a_keys):
        if not stem.startswith(adapter_prefix):
            raise RuntimeError(f"unexpected PEFT tensor prefix: {stem}")
        base_name = stem[len(adapter_prefix) :] + ".weight"
        if base_name not in base:
            raise KeyError(f"adapter target is absent from base checkpoint: {base_name}")

        a = adapter[a_keys[stem]]
        b = adapter[b_keys[stem]]
        weight = base[base_name]
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise TypeError(f"adapter pair for {base_name} is not float32")
        if weight.dtype != torch.float32:
            raise TypeError(f"base tensor {base_name} is not float32")
        if a.shape != (rank, weight.shape[0]):
            raise ValueError(
                f"A shape {tuple(a.shape)} is incompatible with {base_name} "
                f"shape {tuple(weight.shape)}"
            )
        if b.shape != (weight.shape[1], rank):
            raise ValueError(
                f"B shape {tuple(b.shape)} is incompatible with {base_name} "
                f"shape {tuple(weight.shape)}"
            )

        update = (b @ a).transpose(0, 1)
        base[base_name] = (weight + scale * update).contiguous()
        merged_count += 1

    # Required pre-write checks.
    if merged_count != 12:
        raise RuntimeError(f"expected exactly 12 merges, performed {merged_count}")
    if any("lora_" in name for name in base):
        raise RuntimeError("output tensor set unexpectedly contains a LoRA tensor")
    expected_shape = (768, 2304)
    actual_shape = tuple(base["h.0.attn.c_attn.weight"].shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"h.0.attn.c_attn.weight has shape {actual_shape}, expected {expected_shape}"
        )
    if len(base) != 160:
        raise RuntimeError(f"expected 160 output tensors, found {len(base)}")

    shards = pack_shards(base, MAX_SHARD_BYTES)
    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    for shard_number, names in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{total_shards:05d}.safetensors"
        shard = {name: base[name] for name in names}
        shard_bytes = sum(tensor_nbytes(tensor) for tensor in shard.values())
        if shard_bytes > MAX_SHARD_BYTES and len(shard) != 1:
            raise RuntimeError(
                f"multi-tensor shard {filename} exceeds byte limit: {shard_bytes}"
            )
        save_file(shard, OUTPUT_DIR / filename, metadata={"format": "pt"})
        weight_map.update({name: filename for name in names})

    index = {
        "metadata": {"total_size": sum(tensor_nbytes(t) for t in base.values())},
        "weight_map": weight_map,
    }
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Merged {merged_count} LoRA pairs; wrote {len(base)} tensors "
        f"across {total_shards} shards to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
