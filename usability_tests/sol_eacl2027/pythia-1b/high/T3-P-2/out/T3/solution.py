#!/usr/bin/env python3
"""Export Pythia-1B with mixed precision into bounded safetensors shards."""

import json
import os
from math import prod
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T3")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024


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


def fail_unless(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def plan_export() -> tuple[dict[str, torch.dtype], dict[str, tuple[int, ...]], list[list[str]]]:
    """Validate the requested transform and plan shards without writing output."""
    projections = projection_names()
    buffers = buffer_names()

    fail_unless(len(projections) == 64, "internal error: projection set is not 64 tensors")
    fail_unless(len(buffers) == 48, "internal error: buffer set is not 48 tensors")

    with safe_open(INPUT_PATH, framework="pt", device="cpu") as source:
        input_names = set(source.keys())
        missing_projections = sorted(projections - input_names)
        missing_buffers = sorted(buffers - input_names)
        fail_unless(not missing_projections, f"missing projection tensors: {missing_projections}")
        fail_unless(not missing_buffers, f"missing buffers: {missing_buffers}")

        output_names = sorted(input_names - buffers)
        shapes = {
            name: tuple(source.get_slice(name).get_shape())
            for name in output_names
        }

    planned_dtypes = {
        name: torch.bfloat16 if name in projections else torch.float32
        for name in output_names
    }

    # Required checks: all are evaluated before any checkpoint output is written.
    bf16_count = sum(dtype == torch.bfloat16 for dtype in planned_dtypes.values())
    fail_unless(bf16_count == 64, f"expected 64 bfloat16 tensors, planned {bf16_count}")
    qkv0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    fail_unless(planned_dtypes.get(qkv0) == torch.bfloat16, f"{qkv0} is not bfloat16")
    embed = "gpt_neox.embed_in.weight"
    fail_unless(planned_dtypes.get(embed) == torch.float32, f"{embed} is not float32")
    fail_unless(len(planned_dtypes) == 196, f"expected 196 output tensors, planned {len(planned_dtypes)}")
    fail_unless(
        all(dtype == torch.float32 for name, dtype in planned_dtypes.items() if name not in projections),
        "at least one non-projection tensor is not planned as float32",
    )

    element_sizes = {torch.bfloat16: 2, torch.float32: 4}
    tensor_bytes = {
        name: prod(shapes[name]) * element_sizes[planned_dtypes[name]]
        for name in output_names
    }

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in output_names:
        size = tensor_bytes[name]
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = []
                current_bytes = 0
            shards.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current = []
                current_bytes = 0
            current.append(name)
            current_bytes += size
    if current:
        shards.append(current)

    for shard in shards:
        size = sum(tensor_bytes[name] for name in shard)
        fail_unless(
            size <= MAX_SHARD_BYTES or (len(shard) == 1 and tensor_bytes[shard[0]] > MAX_SHARD_BYTES),
            f"invalid shard plan ({size} bytes): {shard}",
        )

    return planned_dtypes, shapes, shards


def main() -> None:
    planned_dtypes, _shapes, shards = plan_export()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear only stale files owned by this exporter, after all required checks pass.
    for stale in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        stale.unlink()
    INDEX_PATH.unlink(missing_ok=True)

    shard_count = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for number, names in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        destination = OUTPUT_DIR / filename
        temporary = OUTPUT_DIR / f".{filename}.tmp"

        with safe_open(INPUT_PATH, framework="pt", device="cpu") as source:
            tensors = {
                name: source.get_tensor(name).to(dtype=planned_dtypes[name]).contiguous()
                for name in names
            }

        actual_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        if actual_bytes > MAX_SHARD_BYTES:
            fail_unless(
                len(tensors) == 1,
                f"shard {filename} contains {actual_bytes} bytes across multiple tensors",
            )
            only_tensor = next(iter(tensors.values()))
            fail_unless(
                only_tensor.numel() * only_tensor.element_size() > MAX_SHARD_BYTES,
                f"oversized shard {filename} does not contain an oversized tensor",
            )

        save_file(tensors, temporary)
        os.replace(temporary, destination)
        for name in names:
            weight_map[name] = filename
        total_size += actual_bytes
        del tensors

    fail_unless(set(weight_map) == set(planned_dtypes), "index weight map does not match output tensors")
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    temporary_index = OUTPUT_DIR / ".model.safetensors.index.json.tmp"
    temporary_index.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_index, INDEX_PATH)

    print(f"Wrote {len(weight_map)} tensors in {shard_count} shards ({total_size} tensor bytes).")


if __name__ == "__main__":
    main()
