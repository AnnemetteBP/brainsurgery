#!/usr/bin/env python3
"""Export the OLMo checkpoint with projection-only BF16 and HF-style sharding."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_DIR = Path("inputs/base")
OUTPUT_DIR = Path("out/T3")
INPUT_INDEX = INPUT_DIR / "model.safetensors.index.json"
OUTPUT_INDEX = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024


def projection_names() -> set[str]:
    attention = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp = ("gate_proj", "up_proj", "down_proj")
    names = {
        f"model.layers.{layer}.self_attn.{projection}.weight"
        for layer in range(16)
        for projection in attention
    }
    names.update(
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(16)
        for projection in mlp
    )
    return names


def pack_shards(names: list[str], output_nbytes: dict[str, int]) -> list[list[str]]:
    """Pack in sorted-name order, except oversize tensors always stand alone."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for name in names:
        size = output_nbytes[name]
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([name])
            continue

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
    with INPUT_INDEX.open("r", encoding="utf-8") as handle:
        input_index = json.load(handle)
    source_map = input_index["weight_map"]
    names = sorted(source_map)
    projections = projection_names()

    # Validate the exact logical output and all source tensors before writing.
    expected_names = projections | {"model.embed_tokens.weight", "lm_head.weight"}
    assert len(projections) == 112, f"expected 112 projection names, got {len(projections)}"
    assert set(names) == expected_names, "input key set is not the expected 114-tensor checkpoint"
    assert len(names) == 114, f"output plan has {len(names)} tensors, expected 114"
    assert "model.layers.0.self_attn.q_proj.weight" in projections
    assert "model.embed_tokens.weight" not in projections

    source_files = sorted(set(source_map.values()))
    for filename in source_files:
        assert (INPUT_DIR / filename).is_file(), f"missing source shard: {filename}"

    output_nbytes: dict[str, int] = {}
    with ExitStack() as stack:
        readers = {
            filename: stack.enter_context(
                safe_open(INPUT_DIR / filename, framework="pt", device="cpu")
            )
            for filename in source_files
        }
        actual_source_map: dict[str, str] = {}
        for filename, reader in readers.items():
            for name in reader.keys():
                assert name not in actual_source_map, f"duplicate source tensor: {name}"
                actual_source_map[name] = filename
        assert actual_source_map == source_map, "input index does not match its shard contents"

        for name in names:
            tensor = readers[source_map[name]].get_tensor(name)
            assert tensor.dtype == torch.float32, f"source tensor is not float32: {name}"
            bytes_per_element = 2 if name in projections else 4
            output_nbytes[name] = tensor.numel() * bytes_per_element

    planned_dtypes = {
        name: (torch.bfloat16 if name in projections else torch.float32)
        for name in names
    }
    assert sum(dtype == torch.bfloat16 for dtype in planned_dtypes.values()) == 112
    assert planned_dtypes["model.layers.0.self_attn.q_proj.weight"] == torch.bfloat16
    assert planned_dtypes["model.embed_tokens.weight"] == torch.float32
    assert len(planned_dtypes) == 114

    shard_groups = pack_shards(names, output_nbytes)
    for group in shard_groups:
        group_size = sum(output_nbytes[name] for name in group)
        assert group_size <= MAX_SHARD_BYTES or len(group) == 1
        if group_size > MAX_SHARD_BYTES:
            assert output_nbytes[group[0]] > MAX_SHARD_BYTES

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_shard in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    if OUTPUT_INDEX.exists():
        OUTPUT_INDEX.unlink()

    shard_count = len(shard_groups)
    output_weight_map: dict[str, str] = {}
    for shard_number, group in enumerate(shard_groups, start=1):
        output_filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
        needed_sources = sorted({source_map[name] for name in group})
        with ExitStack() as stack:
            readers = {
                filename: stack.enter_context(
                    safe_open(INPUT_DIR / filename, framework="pt", device="cpu")
                )
                for filename in needed_sources
            }
            tensors: dict[str, torch.Tensor] = {}
            for name in group:
                tensor = readers[source_map[name]].get_tensor(name)
                if name in projections:
                    tensor = tensor.to(torch.bfloat16)
                else:
                    assert tensor.dtype == torch.float32
                tensors[name] = tensor.contiguous()
            save_file(tensors, OUTPUT_DIR / output_filename, metadata={"format": "pt"})
        for name in group:
            output_weight_map[name] = output_filename

    output_index = {
        "metadata": {"total_size": sum(output_nbytes.values())},
        "weight_map": output_weight_map,
    }
    with OUTPUT_INDEX.open("w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # Verify the written layout, key set, dtypes, and actual tensor-data sizes.
    written_names: set[str] = set()
    written_bfloat16 = 0
    for group in shard_groups:
        filename = output_weight_map[group[0]]
        with safe_open(OUTPUT_DIR / filename, framework="pt", device="cpu") as reader:
            shard_names = set(reader.keys())
            assert shard_names == set(group), f"incorrect contents in {filename}"
            actual_size = 0
            for name in reader.keys():
                tensor = reader.get_tensor(name)
                assert tensor.dtype == planned_dtypes[name], f"incorrect dtype for {name}"
                actual_size += tensor.numel() * tensor.element_size()
                written_bfloat16 += tensor.dtype == torch.bfloat16
            assert actual_size <= MAX_SHARD_BYTES or len(shard_names) == 1
            written_names.update(shard_names)

    assert written_names == expected_names
    assert len(written_names) == 114
    assert written_bfloat16 == 112
    print(
        f"Wrote {len(written_names)} tensors in {shard_count} shards "
        f"({written_bfloat16} bfloat16, {len(written_names) - written_bfloat16} float32)."
    )


if __name__ == "__main__":
    main()
