#!/usr/bin/env python3
"""Export the OLMo checkpoint with projection weights stored as bfloat16."""

import gc
import json
from collections import defaultdict
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


def inspect_sources(weight_map: dict[str, str]) -> dict[str, tuple[list[int], str]]:
    """Read tensor metadata without materializing the checkpoint."""
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        names_by_file[filename].append(name)

    metadata: dict[str, tuple[list[int], str]] = {}
    for filename, expected_names in names_by_file.items():
        with safe_open(INPUT_DIR / filename, framework="pt", device="cpu") as source:
            actual_names = set(source.keys())
            assert actual_names == set(expected_names), (
                f"Index/file key mismatch in {filename}: "
                f"missing={set(expected_names) - actual_names}, "
                f"extra={actual_names - set(expected_names)}"
            )
            for name in expected_names:
                tensor_slice = source.get_slice(name)
                metadata[name] = (tensor_slice.get_shape(), tensor_slice.get_dtype())
    return metadata


def tensor_bytes(shape: list[int], element_size: int) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * element_size


def make_shards(names: list[str], output_bytes: dict[str, int]) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name in names:
        size = output_bytes[name]
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
    return shards


def main() -> None:
    projections = projection_names()
    assert len(projections) == 112, f"Expected 112 projection names, got {len(projections)}"

    with (INPUT_DIR / INDEX_NAME).open() as handle:
        input_index = json.load(handle)
    weight_map = input_index["weight_map"]

    expected_names = projections | {"model.embed_tokens.weight", "lm_head.weight"}
    assert len(weight_map) == 114, f"Expected 114 input tensors, got {len(weight_map)}"
    assert set(weight_map) == expected_names, (
        f"Unexpected input key set: missing={expected_names - set(weight_map)}, "
        f"extra={set(weight_map) - expected_names}"
    )

    source_metadata = inspect_sources(weight_map)
    assert len(source_metadata) == 114, (
        f"Expected metadata for 114 tensors, got {len(source_metadata)}"
    )
    non_float32 = {
        name: dtype for name, (_, dtype) in source_metadata.items() if dtype != "F32"
    }
    assert not non_float32, f"Input tensors must all be float32: {non_float32}"

    # Establish and validate the complete output plan before creating checkpoint files.
    output_dtypes = {
        name: (torch.bfloat16 if name in projections else torch.float32)
        for name in weight_map
    }
    assert sum(dtype == torch.bfloat16 for dtype in output_dtypes.values()) == 112, (
        "Output plan must contain exactly 112 bfloat16 tensors"
    )
    assert (
        output_dtypes["model.layers.0.self_attn.q_proj.weight"] == torch.bfloat16
    ), "Layer 0 q_proj must be bfloat16"
    assert output_dtypes["model.embed_tokens.weight"] == torch.float32, (
        "Embedding tensor must be float32"
    )
    assert len(output_dtypes) == 114, "Output plan must contain exactly 114 tensors"

    output_bytes = {
        name: tensor_bytes(shape, 2 if output_dtypes[name] == torch.bfloat16 else 4)
        for name, (shape, _) in source_metadata.items()
    }
    shards = make_shards(sorted(weight_map), output_bytes)
    for shard in shards:
        size = sum(output_bytes[name] for name in shard)
        assert size <= MAX_SHARD_BYTES or len(shard) == 1, (
            f"Invalid shard plan: {size} bytes across {len(shard)} tensors"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_shard in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        old_shard.unlink()

    output_weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for shard_number, names in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        names_by_source: dict[str, list[str]] = defaultdict(list)
        for name in names:
            names_by_source[weight_map[name]].append(name)

        for source_filename, source_names in names_by_source.items():
            with safe_open(
                INPUT_DIR / source_filename, framework="pt", device="cpu"
            ) as source:
                for name in source_names:
                    tensor = source.get_tensor(name).to(output_dtypes[name])
                    assert tensor.dtype == output_dtypes[name], (
                        f"Wrong converted dtype for {name}: {tensor.dtype}"
                    )
                    tensors[name] = tensor.contiguous()

        assert set(tensors) == set(names), f"Failed to materialize all tensors for {filename}"
        save_file(tensors, OUTPUT_DIR / filename)
        for name in names:
            output_weight_map[name] = filename
        del tensors
        gc.collect()

    assert len(output_weight_map) == 114, (
        f"Expected 114 output index entries, got {len(output_weight_map)}"
    )
    output_index = {
        "metadata": {"total_size": sum(output_bytes.values())},
        "weight_map": output_weight_map,
    }
    with (OUTPUT_DIR / INDEX_NAME).open("w") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"Wrote {len(output_weight_map)} tensors in {shard_count} shards "
        f"({sum(output_bytes.values())} tensor bytes)"
    )


if __name__ == "__main__":
    main()
