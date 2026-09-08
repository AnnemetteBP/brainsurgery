#!/usr/bin/env python3
"""Create the mixed-precision, sharded GPT-2 checkpoint required by T3."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
SHARD_LIMIT = 64 * 1024 * 1024


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Pack tensors in name order without exceeding SHARD_LIMIT.

    A tensor larger than the limit is emitted as a shard by itself.
    """
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_size = 0

    for name in sorted(tensors):
        tensor = tensors[name]
        size = tensor_bytes(tensor)
        if current and current_size + size > SHARD_LIMIT:
            shards.append(current)
            current = {}
            current_size = 0
        if size > SHARD_LIMIT:
            assert not current
            shards.append({name: tensor})
        else:
            current[name] = tensor
            current_size += size

    if current:
        shards.append(current)
    return shards


def main() -> None:
    source = load_file(INPUT, device="cpu")
    projection_names = {
        f"h.{layer}.{module}.weight"
        for layer in range(12)
        for module in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    }
    buffer_names = {f"h.{layer}.attn.bias" for layer in range(12)}

    missing_projections = projection_names - source.keys()
    missing_buffers = buffer_names - source.keys()
    assert not missing_projections, f"missing projection tensors: {sorted(missing_projections)}"
    assert not missing_buffers, f"missing buffers: {sorted(missing_buffers)}"

    result: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        if name in buffer_names:
            continue
        result[name] = tensor.to(torch.bfloat16) if name in projection_names else tensor

    # Required pre-write checks, plus stronger checks for the full dtype policy.
    bf16_names = {name for name, tensor in result.items() if tensor.dtype == torch.bfloat16}
    assert len(bf16_names) == 48, f"expected 48 bfloat16 tensors, got {len(bf16_names)}"
    assert bf16_names == projection_names, "bfloat16 tensors differ from the projection allowlist"
    assert result["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
    assert result["wte.weight"].dtype == torch.float32
    assert len(result) == 148, f"expected 148 output tensors, got {len(result)}"
    assert set(result) == set(source) - buffer_names, "output key set is incorrect"
    non_fp32 = {
        name: str(tensor.dtype)
        for name, tensor in result.items()
        if name not in projection_names and tensor.dtype != torch.float32
    }
    assert not non_fp32, f"non-projection tensors are not float32: {non_fp32}"

    shards = make_shards(result)
    for shard in shards:
        sizes = [tensor_bytes(tensor) for tensor in shard.values()]
        assert sum(sizes) <= SHARD_LIMIT or (
            len(sizes) == 1 and sizes[0] > SHARD_LIMIT
        ), "invalid shard size"

    total_shards = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / filename)
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in result.values())},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (OUTPUT_DIR / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    print(f"Wrote {len(result)} tensors across {total_shards} shards to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
