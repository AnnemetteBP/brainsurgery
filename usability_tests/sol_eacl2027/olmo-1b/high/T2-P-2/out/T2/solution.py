#!/usr/bin/env python3

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE_DIR = Path("inputs/base")
INDEX_PATH = BASE_DIR / "model.safetensors.index.json"
OUTPUT_PATH = Path("out/T2/model.safetensors")

NUM_LAYERS = 16
HIDDEN_SIZE = 2048
HEAD_SIZE = 128
HEAD_TO_REMOVE = 5
EXPECTED_TENSOR_COUNT = 114


def main() -> None:
    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    weight_map = index["weight_map"]
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_TENSOR_COUNT} indexed tensors, got {len(weight_map)}"
        )

    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard = load_file(str(BASE_DIR / shard_name), device="cpu")
        expected_in_shard = {name for name, filename in weight_map.items() if filename == shard_name}
        if set(shard) != expected_in_shard:
            missing = sorted(expected_in_shard - set(shard))
            unexpected = sorted(set(shard) - expected_in_shard)
            raise AssertionError(
                f"Index mismatch in {shard_name}: missing={missing}, unexpected={unexpected}"
            )
        overlap = tensors.keys() & shard.keys()
        if overlap:
            raise AssertionError(f"Duplicate tensor names across shards: {sorted(overlap)}")
        tensors.update(shard)

    if set(tensors) != set(weight_map):
        raise AssertionError("Loaded tensor names do not exactly match the index")

    cut_start = HEAD_TO_REMOVE * HEAD_SIZE
    cut_end = cut_start + HEAD_SIZE

    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.{projection}.weight"
            source = tensors[name]
            if list(source.shape) != [HIDDEN_SIZE, HIDDEN_SIZE]:
                raise AssertionError(f"Unexpected source shape for {name}: {list(source.shape)}")
            tensors[name] = torch.cat(
                (source[:cut_start, :], source[cut_end:, :]), dim=0
            )

        name = f"{prefix}.o_proj.weight"
        source = tensors[name]
        if list(source.shape) != [HIDDEN_SIZE, HIDDEN_SIZE]:
            raise AssertionError(f"Unexpected source shape for {name}: {list(source.shape)}")
        tensors[name] = torch.cat(
            (source[:, :cut_start], source[:, cut_end:]), dim=1
        )

    # Validate all transformed tensors before writing the checkpoint.
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.{projection}.weight"
            if list(tensors[name].shape) != [1920, 2048]:
                raise AssertionError(f"Incorrect output shape for {name}: {list(tensors[name].shape)}")

        name = f"{prefix}.o_proj.weight"
        if list(tensors[name].shape) != [2048, 1920]:
            raise AssertionError(f"Incorrect output shape for {name}: {list(tensors[name].shape)}")

    # Explicit task-required layer-0 checks.
    required_shapes = {
        "model.layers.0.self_attn.q_proj.weight": [1920, 2048],
        "model.layers.0.self_attn.k_proj.weight": [1920, 2048],
        "model.layers.0.self_attn.v_proj.weight": [1920, 2048],
        "model.layers.0.self_attn.o_proj.weight": [2048, 1920],
    }
    for name, expected_shape in required_shapes.items():
        if list(tensors[name].shape) != expected_shape:
            raise AssertionError(
                f"Required check failed for {name}: expected {expected_shape}, "
                f"got {list(tensors[name].shape)}"
            )
    if len(tensors) != EXPECTED_TENSOR_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_TENSOR_COUNT} output tensors, got {len(tensors)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUTPUT_PATH))
    print(f"Wrote {len(tensors)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
