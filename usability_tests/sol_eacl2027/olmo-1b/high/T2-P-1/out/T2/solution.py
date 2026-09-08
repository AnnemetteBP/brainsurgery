#!/usr/bin/env python3
"""Remove attention head 5 from every OLMo transformer layer."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


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

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"Missing or invalid weight_map in {INDEX_PATH}")
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TENSOR_COUNT} indexed tensors, got {len(weight_map)}"
        )

    tensors_by_shard: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_name in weight_map.items():
        tensors_by_shard[shard_name].append(tensor_name)

    row_projection_names = {
        f"model.layers.{layer}.self_attn.{projection}.weight"
        for layer in range(NUM_LAYERS)
        for projection in ("q_proj", "k_proj", "v_proj")
    }
    column_projection_names = {
        f"model.layers.{layer}.self_attn.o_proj.weight"
        for layer in range(NUM_LAYERS)
    }
    projection_names = row_projection_names | column_projection_names

    head_start = HEAD_TO_REMOVE * HEAD_SIZE
    head_end = head_start + HEAD_SIZE
    output_tensors: dict[str, torch.Tensor] = {}

    for shard_name, tensor_names in sorted(tensors_by_shard.items()):
        shard_path = BASE_DIR / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Checkpoint shard not found: {shard_path}")

        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            actual_names = set(shard.keys())
            indexed_names = set(tensor_names)
            if actual_names != indexed_names:
                missing = sorted(indexed_names - actual_names)
                unexpected = sorted(actual_names - indexed_names)
                raise RuntimeError(
                    f"Shard/index key mismatch for {shard_name}: "
                    f"missing={missing}, unexpected={unexpected}"
                )

            for tensor_name in tensor_names:
                tensor = shard.get_tensor(tensor_name)
                if tensor_name in projection_names:
                    if tuple(tensor.shape) != (HIDDEN_SIZE, HIDDEN_SIZE):
                        raise RuntimeError(
                            f"Unexpected source shape for {tensor_name}: "
                            f"{list(tensor.shape)}"
                        )
                    if tensor_name in row_projection_names:
                        tensor = torch.cat(
                            (tensor[:head_start, :], tensor[head_end:, :]), dim=0
                        )
                    else:
                        tensor = torch.cat(
                            (tensor[:, :head_start], tensor[:, head_end:]), dim=1
                        )
                output_tensors[tensor_name] = tensor

    if set(output_tensors) != set(weight_map):
        raise RuntimeError("Output tensor names do not exactly match the checkpoint index")

    expected_projection_shapes = {
        **{name: (1920, 2048) for name in row_projection_names},
        **{name: (2048, 1920) for name in column_projection_names},
    }
    for tensor_name, expected_shape in expected_projection_shapes.items():
        actual_shape = tuple(output_tensors[tensor_name].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Incorrect output shape for {tensor_name}: "
                f"expected {list(expected_shape)}, got {list(actual_shape)}"
            )

    # Explicit task-required checks, all performed before writing.
    required_shapes = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for tensor_name, expected_shape in required_shapes.items():
        if tuple(output_tensors[tensor_name].shape) != expected_shape:
            raise RuntimeError(
                f"Required check failed for {tensor_name}: "
                f"expected {list(expected_shape)}, "
                f"got {list(output_tensors[tensor_name].shape)}"
            )
    if len(output_tensors) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"Required check failed: expected {EXPECTED_TENSOR_COUNT} tensors, "
            f"got {len(output_tensors)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, OUTPUT_PATH)
    print(f"Wrote {len(output_tensors)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
