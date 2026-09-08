#!/usr/bin/env python3
"""Export the supplied Pythia checkpoint with exact mixed dtypes and sharding."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
INDEX = OUTPUT / "model.safetensors.index.json"
SHARD_LIMIT = 256 * 1024 * 1024
LAYERS = range(16)
PROJECTION_SUFFIXES = (
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
)
BUFFER_SUFFIXES = (
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
)


def names_for(suffixes: tuple[str, ...]) -> set[str]:
    return {f"gpt_neox.layers.{layer}.{suffix}" for layer in LAYERS for suffix in suffixes}


def tensor_bytes(shape: list[int], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def main() -> None:
    projections = names_for(PROJECTION_SUFFIXES)
    buffers = names_for(BUFFER_SUFFIXES)

    # Construct and validate the complete output plan before writing any checkpoint file.
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        missing_projections = projections - source_keys
        missing_buffers = buffers - source_keys
        assert not missing_projections, f"missing projection tensors: {sorted(missing_projections)}"
        assert not missing_buffers, f"missing buffers: {sorted(missing_buffers)}"
        assert len(source_keys) == 244, f"expected 244 input tensors, found {len(source_keys)}"

        output_keys = sorted(source_keys - buffers)
        target_dtypes = {
            name: torch.bfloat16 if name in projections else torch.float32
            for name in output_keys
        }
        shapes = {name: source.get_slice(name).get_shape() for name in output_keys}

        # Required checks: these deliberately run before any checkpoint output is written.
        assert sum(dtype == torch.bfloat16 for dtype in target_dtypes.values()) == 64
        assert target_dtypes["gpt_neox.layers.0.attention.query_key_value.weight"] == torch.bfloat16
        assert target_dtypes["gpt_neox.embed_in.weight"] == torch.float32
        assert len(output_keys) == 196, f"expected 196 output tensors, found {len(output_keys)}"
        assert set(output_keys) == source_keys - buffers
        assert all(
            source.get_slice(name).get_dtype() in {"F16", "F32", "BF16"}
            for name in output_keys
        ), "an output tensor has an unexpected non-floating source dtype"

    sizes = {name: tensor_bytes(shapes[name], target_dtypes[name]) for name in output_keys}
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in output_keys:
        size = sizes[name]
        if size > SHARD_LIMIT:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([name])
        else:
            if current and current_size + size > SHARD_LIMIT:
                shards.append(current)
                current, current_size = [], 0
            current.append(name)
            current_size += size
    if current:
        shards.append(current)

    assert all(
        sum(sizes[name] for name in shard) <= SHARD_LIMIT
        or (len(shard) == 1 and sizes[shard[0]] > SHARD_LIMIT)
        for shard in shards
    )

    # Remove only checkpoint products from a possible earlier invocation.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old_shard in OUTPUT.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    INDEX.unlink(missing_ok=True)

    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        for number, names in enumerate(shards, start=1):
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors = {
                name: source.get_tensor(name).to(dtype=target_dtypes[name]).contiguous()
                for name in names
            }
            actual_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
            planned_size = sum(sizes[name] for name in names)
            assert actual_size == planned_size
            assert actual_size <= SHARD_LIMIT or (len(names) == 1 and actual_size > SHARD_LIMIT)
            save_file(tensors, OUTPUT / filename, metadata={"format": "pt"})
            weight_map.update({name: filename for name in names})

    assert set(weight_map) == set(output_keys)
    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    INDEX.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(output_keys)} tensors in {shard_count} shards "
        f"({sum(sizes.values()):,} tensor bytes)"
    )


if __name__ == "__main__":
    main()
