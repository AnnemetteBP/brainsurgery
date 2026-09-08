#!/usr/bin/env python3
"""Export the OLMo checkpoint with projection-only bf16 and 256 MiB shards."""

import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_DIR = Path("inputs/base")
OUTPUT_DIR = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024


def projection_names() -> set[str]:
    names: set[str] = set()
    for layer in range(16):
        prefix = f"model.layers.{layer}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"{prefix}.self_attn.{projection}.weight")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"{prefix}.mlp.{projection}.weight")
    return names


def main() -> None:
    with (INPUT_DIR / INDEX_NAME).open(encoding="utf-8") as stream:
        source_index = json.load(stream)
    source_map: dict[str, str] = source_index["weight_map"]
    names = sorted(source_map)
    bf16_names = projection_names()

    # Inspect source tensor metadata without materializing the full checkpoint.
    shapes: dict[str, tuple[int, ...]] = {}
    source_dtypes: dict[str, str] = {}
    for filename in sorted(set(source_map.values())):
        expected_here = {name for name, shard in source_map.items() if shard == filename}
        with safe_open(INPUT_DIR / filename, framework="pt", device="cpu") as handle:
            actual_here = set(handle.keys())
            if actual_here != expected_here:
                raise AssertionError(f"source index disagrees with {filename}")
            for name in actual_here:
                tensor_slice = handle.get_slice(name)
                shapes[name] = tuple(tensor_slice.get_shape())
                source_dtypes[name] = tensor_slice.get_dtype()

    # All required checks happen before the first output checkpoint file is written.
    if len(names) != 114:
        raise AssertionError(f"expected 114 output tensors, found {len(names)}")
    if len(bf16_names) != 112 or not bf16_names <= set(names):
        missing = sorted(bf16_names - set(names))
        raise AssertionError(f"expected exactly 112 projection tensors; missing={missing}")
    output_dtypes = {
        name: torch.bfloat16 if name in bf16_names else torch.float32 for name in names
    }
    if sum(dtype == torch.bfloat16 for dtype in output_dtypes.values()) != 112:
        raise AssertionError("output plan does not contain exactly 112 bfloat16 tensors")
    if output_dtypes.get("model.layers.0.self_attn.q_proj.weight") != torch.bfloat16:
        raise AssertionError("layer 0 q_proj is not planned as bfloat16")
    if output_dtypes.get("model.embed_tokens.weight") != torch.float32:
        raise AssertionError("embedding is not planned as float32")
    non_float32 = sorted(name for name, dtype in source_dtypes.items() if dtype != "F32")
    if non_float32:
        raise AssertionError(f"input tensors are not all float32: {non_float32}")

    output_sizes = {
        name: math.prod(shapes[name]) * (2 if name in bf16_names else 4) for name in names
    }

    # Greedy deterministic packing. A tensor over the limit is necessarily alone.
    planned_shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = output_sizes[name]
        if current and (size > MAX_SHARD_BYTES or current_bytes + size > MAX_SHARD_BYTES):
            planned_shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            planned_shards.append(current)
            current = []
            current_bytes = 0
    if current:
        planned_shards.append(current)

    shard_count = len(planned_shards)
    weight_map: dict[str, str] = {}
    for shard_number, shard_names in enumerate(planned_shards, start=1):
        filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        with ExitStack() as stack:
            handles = {
                source_file: stack.enter_context(
                    safe_open(INPUT_DIR / source_file, framework="pt", device="cpu")
                )
                for source_file in {source_map[name] for name in shard_names}
            }
            for name in shard_names:
                source = handles[source_map[name]].get_tensor(name)
                tensor = source.to(torch.bfloat16) if name in bf16_names else source
                tensors[name] = tensor.contiguous()
            save_file(tensors, OUTPUT_DIR / filename, metadata={"format": "pt"})
        for name in shard_names:
            weight_map[name] = filename

    output_index = {
        "metadata": {"total_size": sum(output_sizes.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT_DIR / INDEX_NAME).open("w", encoding="utf-8") as stream:
        json.dump(output_index, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print(
        f"Wrote {len(names)} tensors ({len(bf16_names)} bf16) "
        f"across {shard_count} shards to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
