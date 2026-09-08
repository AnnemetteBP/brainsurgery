#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA adapter into GPT-2 and export shards."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUTPUT_DIR = Path("out/T5")
MAX_SHARD_BYTES = 100 * 1024 * 1024
ADAPTER_PREFIX = "base_model.model."
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Pack tensors in name order without exceeding the raw-data limit."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in sorted(tensors):
        tensor = tensors[name]
        size = tensor_nbytes(tensor)
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            # An oversized tensor is permitted only as a shard by itself.
            assert len(current) == 1
            shards.append(current)
            current = {}
            current_bytes = 0

    if current:
        shards.append(current)
    return shards


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rank = config["r"]
    alpha = config["lora_alpha"]
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"invalid LoRA rank: {rank!r}")
    if config.get("fan_in_fan_out") is not True:
        raise ValueError("this merge expects fan_in_fan_out=true")
    scale = alpha / rank

    base = load_file(str(BASE_PATH), device="cpu")
    adapter = load_file(str(ADAPTER_PATH), device="cpu")
    output = dict(base)

    a_names = {name for name in adapter if name.endswith(A_SUFFIX)}
    b_names = {name for name in adapter if name.endswith(B_SUFFIX)}
    if len(a_names) != 12 or len(b_names) != 12:
        raise RuntimeError(
            f"expected 12 A and 12 B tensors, found {len(a_names)} and {len(b_names)}"
        )

    merged = 0
    consumed: set[str] = set()
    for a_name in sorted(a_names):
        stem = a_name[: -len(A_SUFFIX)]
        b_name = stem + B_SUFFIX
        if b_name not in adapter:
            raise KeyError(f"missing adapter pair for {a_name}")
        if not stem.startswith(ADAPTER_PREFIX):
            raise ValueError(f"unexpected adapter name: {a_name}")
        base_name = stem[len(ADAPTER_PREFIX) :] + ".weight"
        if base_name not in output:
            raise KeyError(f"adapter target absent from base: {base_name}")

        a = adapter[a_name]
        b = adapter[b_name]
        target = output[base_name]
        if a.dtype != torch.float32 or b.dtype != torch.float32 or target.dtype != torch.float32:
            raise TypeError(f"non-float32 merge inputs for {base_name}")
        if a.shape != (rank, target.shape[0]) or b.shape != (target.shape[1], rank):
            raise ValueError(
                f"incompatible shapes for {base_name}: A={tuple(a.shape)}, "
                f"B={tuple(b.shape)}, base={tuple(target.shape)}"
            )

        delta = (b @ a).transpose(0, 1)
        output[base_name] = target + scale * delta
        consumed.update((a_name, b_name))
        merged += 1

    if consumed != set(adapter):
        raise RuntimeError(f"unconsumed adapter tensors: {sorted(set(adapter) - consumed)}")

    # Required pre-write checks.
    if merged != 12:
        raise RuntimeError(f"expected exactly 12 merged pairs, got {merged}")
    if any("lora_" in name for name in output):
        raise RuntimeError("LoRA tensor leaked into dense output")
    if output["h.0.attn.c_attn.weight"].shape != (768, 2304):
        raise RuntimeError("h.0.attn.c_attn.weight has the wrong shape")
    if len(output) != 160:
        raise RuntimeError(f"expected 160 output tensors, got {len(output)}")
    if set(output) != set(base):
        raise RuntimeError("output key set differs from base key set")

    shards = make_shards(output)
    for shard in shards:
        sizes = [tensor_nbytes(tensor) for tensor in shard.values()]
        if sum(sizes) > MAX_SHARD_BYTES and len(shard) != 1:
            raise RuntimeError("multi-tensor shard exceeds 100 MiB")

    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, str(OUTPUT_DIR / filename))
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_nbytes(t) for t in output.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT_DIR / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Merged {merged} LoRA pairs and wrote {len(output)} tensors in {total_shards} shards")


if __name__ == "__main__":
    main()
