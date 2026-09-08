#!/usr/bin/env python3
"""Merge the supplied LoRA adapter into GPT-2 and export safetensor shards."""

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


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Pack whole tensors in sorted-name order, honoring the shard byte cap."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in sorted(tensors):
        tensor = tensors[name]
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
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rank = config["r"]
    alpha = config["lora_alpha"]
    assert isinstance(rank, int) and rank > 0, f"Invalid LoRA rank: {rank!r}"
    assert config.get("fan_in_fan_out") is True, "Expected fan_in_fan_out=true"
    scale = float(alpha) / rank

    base = load_file(BASE_PATH, device="cpu")
    adapter = load_file(ADAPTER_PATH, device="cpu")
    output = dict(base)

    a_names = sorted(name for name in adapter if name.endswith(A_SUFFIX))
    b_names = sorted(name for name in adapter if name.endswith(B_SUFFIX))
    assert len(a_names) == 12, f"Expected 12 LoRA A tensors, found {len(a_names)}"
    assert len(b_names) == 12, f"Expected 12 LoRA B tensors, found {len(b_names)}"

    merged = 0
    paired_adapter_names: set[str] = set()
    for a_name in a_names:
        stem = a_name[: -len(A_SUFFIX)]
        b_name = stem + B_SUFFIX
        assert b_name in adapter, f"Missing LoRA B tensor paired with {a_name}"
        assert stem.startswith(ADAPTER_PREFIX), f"Unexpected adapter name: {a_name}"
        base_name = stem[len(ADAPTER_PREFIX) :] + ".weight"
        assert base_name in output, f"Missing base tensor for {a_name}: {base_name}"

        a = adapter[a_name]
        b = adapter[b_name]
        weight = output[base_name]
        assert a.dtype == torch.float32 and b.dtype == torch.float32, (
            f"Adapter pair for {base_name} is not float32"
        )
        assert weight.dtype == torch.float32, f"Base tensor {base_name} is not float32"
        assert a.ndim == 2 and b.ndim == 2, f"LoRA factors for {base_name} must be matrices"
        assert a.shape[0] == rank and b.shape[1] == rank, (
            f"LoRA rank mismatch for {base_name}: A={tuple(a.shape)}, B={tuple(b.shape)}"
        )

        delta = (b @ a).T
        assert delta.shape == weight.shape, (
            f"Merged update shape mismatch for {base_name}: "
            f"delta={tuple(delta.shape)}, base={tuple(weight.shape)}"
        )
        output[base_name] = weight + delta * scale
        paired_adapter_names.update((a_name, b_name))
        merged += 1

    assert paired_adapter_names == set(adapter), "Adapter contains unpaired or unexpected tensors"

    # Required pre-write checks.
    assert merged == 12, f"Expected exactly 12 merged adapter pairs, found {merged}"
    assert not any("lora_" in name for name in output), "LoRA tensor leaked into output"
    assert output["h.0.attn.c_attn.weight"].shape == (768, 2304), (
        "h.0.attn.c_attn.weight has the wrong shape"
    )
    assert len(output) == 160, f"Expected 160 output tensors, found {len(output)}"
    assert set(output) == set(base), "Output tensor names differ from the base checkpoint"

    shards = make_shards(output)
    assert all(
        sum(tensor_bytes(tensor) for tensor in shard.values()) <= MAX_SHARD_BYTES
        or (len(shard) == 1 and tensor_bytes(next(iter(shard.values()))) > MAX_SHARD_BYTES)
        for shard in shards
    ), "Internal error: invalid shard packing"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / filename, metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in output.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT_DIR / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Merged {merged} LoRA pairs and wrote {len(output)} tensors to {total_shards} shards")


if __name__ == "__main__":
    main()
