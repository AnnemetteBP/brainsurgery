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
MAX_SHARD_BYTES = 256 * 1024 * 1024

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


def expected_names() -> tuple[set[str], set[str]]:
    projections = {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in PROJECTION_SUFFIXES
    }
    buffers = {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in BUFFER_SUFFIXES
    }
    return projections, buffers


def tensor_bytes(shape: list[int], dtype: torch.dtype) -> int:
    return math.prod(shape) * torch.empty((), dtype=dtype).element_size()


def build_plan() -> tuple[list[list[str]], dict[str, torch.dtype], int]:
    projections, buffers = expected_names()
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        input_names = set(source.keys())
        missing_projections = projections - input_names
        missing_buffers = buffers - input_names
        assert not missing_projections, f"missing projection tensors: {sorted(missing_projections)}"
        assert not missing_buffers, f"missing buffers: {sorted(missing_buffers)}"

        output_names = sorted(input_names - buffers)
        planned_dtype = {
            name: torch.bfloat16 if name in projections else torch.float32
            for name in output_names
        }
        sizes = {
            name: tensor_bytes(source.get_slice(name).get_shape(), planned_dtype[name])
            for name in output_names
        }

    # Required checks are deliberately complete before any output shard is written.
    assert len(projections) == 64
    assert sum(dtype == torch.bfloat16 for dtype in planned_dtype.values()) == 64
    assert (
        planned_dtype["gpt_neox.layers.0.attention.query_key_value.weight"]
        == torch.bfloat16
    )
    assert planned_dtype["gpt_neox.embed_in.weight"] == torch.float32
    assert len(output_names) == 196
    assert not (buffers & set(output_names))
    assert set(planned_dtype) == input_names - buffers

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in output_names:
        size = sizes[name]
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
    if current:
        shards.append(current)

    for shard in shards:
        size = sum(sizes[name] for name in shard)
        assert size <= MAX_SHARD_BYTES or (len(shard) == 1 and size > MAX_SHARD_BYTES)
    return shards, planned_dtype, sum(sizes.values())


def export() -> None:
    shards, planned_dtype, total_size = build_plan()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    existing = list(OUTPUT.glob("model-*-of-*.safetensors"))
    assert not existing and not INDEX.exists(), "refusing to overwrite an existing export"

    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        for number, names in enumerate(shards, start=1):
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for name in names:
                converted = source.get_tensor(name).to(planned_dtype[name])
                assert converted.dtype == planned_dtype[name]
                tensors[name] = converted.contiguous()
                weight_map[name] = filename
            save_file(tensors, OUTPUT / filename, metadata={"format": "pt"})

    assert set(weight_map) == set(planned_dtype)
    index_data = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    INDEX.write_text(json.dumps(index_data, indent=2, sort_keys=True) + "\n")
    verify_export(shards, planned_dtype, weight_map)


def verify_export(
    shards: list[list[str]],
    planned_dtype: dict[str, torch.dtype],
    weight_map: dict[str, str],
) -> None:
    seen: set[str] = set()
    bf16_count = 0
    for names in shards:
        filename = weight_map[names[0]]
        with safe_open(OUTPUT / filename, framework="pt", device="cpu") as shard:
            actual_names = set(shard.keys())
            assert actual_names == set(names), f"wrong keys in {filename}"
            shard_bytes = 0
            for name in actual_names:
                tensor = shard.get_tensor(name)
                assert tensor.dtype == planned_dtype[name], f"wrong dtype for {name}"
                bf16_count += tensor.dtype == torch.bfloat16
                shard_bytes += tensor.numel() * tensor.element_size()
            assert shard_bytes <= MAX_SHARD_BYTES or (
                len(actual_names) == 1 and shard_bytes > MAX_SHARD_BYTES
            ), f"oversized multi-tensor shard: {filename}"
            seen.update(actual_names)

    assert len(seen) == 196
    assert bf16_count == 64
    assert "gpt_neox.layers.0.attention.query_key_value.weight" in seen
    assert "gpt_neox.embed_in.weight" in seen
    print(f"Export verified: {len(seen)} tensors, {bf16_count} bfloat16, {len(shards)} shards")


if __name__ == "__main__":
    export()
