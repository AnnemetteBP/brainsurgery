#!/usr/bin/env python3
"""Export the task checkpoint with exact mixed dtypes and bounded shards."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
INDEX = OUTPUT / "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024
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
PROJECTIONS = {
    f"gpt_neox.layers.{layer}.{suffix}"
    for layer in LAYERS
    for suffix in PROJECTION_SUFFIXES
}
BUFFERS = {
    f"gpt_neox.layers.{layer}.{suffix}"
    for layer in LAYERS
    for suffix in BUFFER_SUFFIXES
}


def numel(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def inspect_and_plan() -> tuple[list[list[str]], dict[str, int]]:
    """Validate all required preconditions and make a deterministic shard plan."""
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        source_key_set = set(source_keys)

        assert len(source_keys) == 244, f"expected 244 source tensors, got {len(source_keys)}"
        assert PROJECTIONS <= source_key_set, "one or more projection matrices are missing"
        assert BUFFERS <= source_key_set, "one or more named buffers are missing"
        assert len(PROJECTIONS) == 64, f"expected 64 projection names, got {len(PROJECTIONS)}"
        assert len(BUFFERS) == 48, f"expected 48 buffer names, got {len(BUFFERS)}"

        output_keys = [key for key in source_keys if key not in BUFFERS]
        target_dtypes = {
            key: torch.bfloat16 if key in PROJECTIONS else torch.float32
            for key in output_keys
        }

        # Required checks: these all run before any output checkpoint file is written.
        bf16_count = sum(dtype == torch.bfloat16 for dtype in target_dtypes.values())
        assert bf16_count == 64, f"expected exactly 64 bfloat16 tensors, got {bf16_count}"
        assert (
            target_dtypes["gpt_neox.layers.0.attention.query_key_value.weight"]
            == torch.bfloat16
        ), "layer 0 query_key_value weight is not planned as bfloat16"
        assert target_dtypes["gpt_neox.embed_in.weight"] == torch.float32, (
            "gpt_neox.embed_in.weight is not planned as float32"
        )
        assert len(output_keys) == 196, f"expected exactly 196 output tensors, got {len(output_keys)}"

        sizes: dict[str, int] = {}
        for key in output_keys:
            tensor_slice = source.get_slice(key)
            assert tensor_slice.get_dtype() == "F16", f"unexpected source dtype for {key}"
            element_size = 2 if target_dtypes[key] == torch.bfloat16 else 4
            sizes[key] = numel(tensor_slice.get_shape()) * element_size

    # Greedy, deterministic packing. Oversized tensors are legal only as singleton shards.
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in output_keys:
        size = sizes[key]
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_size = [], 0
            shards.append([key])
        else:
            if current and current_size + size > MAX_SHARD_BYTES:
                shards.append(current)
                current, current_size = [], 0
            current.append(key)
            current_size += size
    if current:
        shards.append(current)

    for shard in shards:
        shard_size = sum(sizes[key] for key in shard)
        assert shard_size <= MAX_SHARD_BYTES or (
            len(shard) == 1 and sizes[shard[0]] > MAX_SHARD_BYTES
        ), f"invalid shard plan: {shard_size} bytes across {len(shard)} tensors"
    return shards, sizes


def export(shards: list[list[str]], sizes: dict[str, int]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT.glob("model-*-of-*.safetensors"):
        stale.unlink()
    if INDEX.exists():
        INDEX.unlink()

    shard_count = len(shards)
    weight_map: OrderedDict[str, str] = OrderedDict()
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        for shard_number, keys in enumerate(shards, start=1):
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
            for key in keys:
                target_dtype = torch.bfloat16 if key in PROJECTIONS else torch.float32
                tensors[key] = source.get_tensor(key).to(dtype=target_dtype).contiguous()
                weight_map[key] = filename
            temporary = OUTPUT / f".{filename}.tmp"
            save_file(tensors, temporary, metadata={"format": "pt"})
            os.replace(temporary, OUTPUT / filename)

    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    temporary_index = OUTPUT / ".model.safetensors.index.json.tmp"
    temporary_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_index, INDEX)


def verify(shards: list[list[str]], sizes: dict[str, int]) -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    assert len(weight_map) == 196, f"index maps {len(weight_map)} tensors, not 196"
    assert index["metadata"]["total_size"] == sum(sizes.values())

    seen: set[str] = set()
    bf16_count = 0
    with safe_open(INPUT, framework="pt", device="cpu") as source:
        for keys in shards:
            filename = weight_map[keys[0]]
            assert all(weight_map[key] == filename for key in keys)
            with safe_open(OUTPUT / filename, framework="pt", device="cpu") as saved:
                saved_keys = set(saved.keys())
                assert saved_keys == set(keys), f"incorrect key set in {filename}"
                payload_size = 0
                for key in saved.keys():
                    tensor = saved.get_tensor(key)
                    expected_dtype = torch.bfloat16 if key in PROJECTIONS else torch.float32
                    assert tensor.dtype == expected_dtype, f"wrong dtype for {key}"
                    expected = source.get_tensor(key).to(expected_dtype)
                    assert torch.equal(tensor, expected), f"value mismatch for {key}"
                    payload_size += tensor.numel() * tensor.element_size()
                    bf16_count += tensor.dtype == torch.bfloat16
                assert payload_size <= MAX_SHARD_BYTES or len(saved_keys) == 1, (
                    f"oversized non-singleton shard {filename}: {payload_size} bytes"
                )
                seen.update(saved_keys)

    assert seen == set(weight_map), "index and shard tensor names differ"
    assert not (seen & BUFFERS), "a deleted buffer is present"
    assert bf16_count == 64, f"found {bf16_count} bfloat16 tensors after saving"
    assert weight_map["gpt_neox.layers.0.attention.query_key_value.weight"]
    assert weight_map["gpt_neox.embed_in.weight"]
    print(
        f"Wrote and verified {len(seen)} tensors in {len(shards)} shards "
        f"({bf16_count} bfloat16, {len(seen) - bf16_count} float32)."
    )


def main() -> None:
    shards, sizes = inspect_and_plan()
    export(shards, sizes)
    verify(shards, sizes)


if __name__ == "__main__":
    main()
