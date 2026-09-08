#!/usr/bin/env python3
"""Export Pythia-1B as a mixed-precision sharded safetensors checkpoint."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MAX_SHARD_BYTES = 256 * 1024 * 1024
OUTPUT_TENSOR_COUNT = 196

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "inputs" / "base" / "model.safetensors"
OUTPUT_DIR = Path(__file__).resolve().parent
INDEX_FILE = OUTPUT_DIR / "model.safetensors.index.json"


def projection_names() -> set[str]:
    suffixes = (
        "attention.query_key_value.weight",
        "attention.dense.weight",
        "mlp.dense_h_to_4h.weight",
        "mlp.dense_4h_to_h.weight",
    )
    return {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in suffixes
    }


def buffer_names() -> set[str]:
    suffixes = (
        "attention.bias",
        "attention.masked_bias",
        "attention.rotary_emb.inv_freq",
    )
    return {
        f"gpt_neox.layers.{layer}.{suffix}"
        for layer in range(16)
        for suffix in suffixes
    }


def make_shard_plan(
    names: list[str], tensor_sizes: dict[str, int]
) -> list[list[str]]:
    """Greedily pack tensors, leaving an oversized tensor alone as required."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for name in names:
        size = tensor_sizes[name]
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([name])
        else:
            if current and current_size + size > MAX_SHARD_BYTES:
                shards.append(current)
                current = []
                current_size = 0
            current.append(name)
            current_size += size

    if current:
        shards.append(current)
    return shards


def main() -> None:
    projections = projection_names()
    buffers = buffer_names()
    assert len(projections) == 64, "internal error: projection allowlist is not 64 names"
    assert len(buffers) == 48, "internal error: buffer denylist is not 48 names"

    # Preflight only reads safetensors metadata. All required checks and the
    # complete shard plan are validated before any checkpoint output is written.
    with safe_open(INPUT_FILE, framework="pt", device="cpu") as source:
        input_names = set(source.keys())
        missing_projections = projections - input_names
        missing_buffers = buffers - input_names
        assert not missing_projections, (
            f"input is missing projection tensors: {sorted(missing_projections)}"
        )
        assert not missing_buffers, f"input is missing buffers: {sorted(missing_buffers)}"

        output_names = sorted(input_names - buffers)
        output_dtypes = {
            name: torch.bfloat16 if name in projections else torch.float32
            for name in output_names
        }
        tensor_sizes = {
            name: math.prod(source.get_slice(name).get_shape())
            * torch.empty((), dtype=output_dtypes[name]).element_size()
            for name in output_names
        }

    # Task-required checks: intentionally performed before writing any shard.
    bf16_count = sum(dtype == torch.bfloat16 for dtype in output_dtypes.values())
    assert bf16_count == 64, f"expected 64 bfloat16 tensors, found {bf16_count}"
    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    assert output_dtypes.get(qkv0) == torch.bfloat16, f"{qkv0} is not bfloat16"
    assert output_dtypes.get("gpt_neox.embed_in.weight") == torch.float32, (
        "gpt_neox.embed_in.weight is not float32"
    )
    assert len(output_names) == OUTPUT_TENSOR_COUNT, (
        f"expected {OUTPUT_TENSOR_COUNT} output tensors, found {len(output_names)}"
    )

    shards = make_shard_plan(output_names, tensor_sizes)
    assert shards, "shard plan is empty"
    assert {name for shard in shards for name in shard} == set(output_names), (
        "shard plan does not contain exactly the output tensors"
    )
    for shard in shards:
        shard_size = sum(tensor_sizes[name] for name in shard)
        assert shard_size <= MAX_SHARD_BYTES or (
            len(shard) == 1 and tensor_sizes[shard[0]] > MAX_SHARD_BYTES
        ), f"invalid shard plan: {shard_size} tensor bytes in one shard"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        stale.unlink()
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()

    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    with safe_open(INPUT_FILE, framework="pt", device="cpu") as source:
        for number, shard_names in enumerate(shards, start=1):
            filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for name in shard_names:
                tensor = source.get_tensor(name).to(output_dtypes[name]).contiguous()
                assert tensor.dtype == output_dtypes[name], (
                    f"dtype conversion failed for {name}: got {tensor.dtype}"
                )
                assert tensor.numel() * tensor.element_size() == tensor_sizes[name], (
                    f"size changed unexpectedly for {name}"
                )
                tensors[name] = tensor
                weight_map[name] = filename
            save_file(tensors, OUTPUT_DIR / filename)

    assert set(weight_map) == set(output_names), "index weight_map is incomplete"
    index = {
        "metadata": {"total_size": sum(tensor_sizes.values())},
        "weight_map": weight_map,
    }
    temporary_index = INDEX_FILE.with_suffix(INDEX_FILE.suffix + ".tmp")
    with temporary_index.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_index, INDEX_FILE)

    print(
        f"Wrote {len(output_names)} tensors in {shard_count} shards "
        f"({bf16_count} bfloat16, {len(output_names) - bf16_count} float32)."
    )


if __name__ == "__main__":
    main()
