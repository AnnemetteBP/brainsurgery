#!/usr/bin/env python3
"""Remove attention head 5 from every OLMo attention projection."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("inputs/base")
INDEX = BASE / "model.safetensors.index.json"
OUTPUT = Path("out/T2/model.safetensors")

LAYERS = 16
HEAD_DIM = 128
REMOVED_HEAD = 5
HIDDEN_SIZE = 2048
PRUNED_SIZE = HIDDEN_SIZE - HEAD_DIM
EXPECTED_TENSORS = 114

ATTENTION_KEY = re.compile(
    r"model\.layers\.(\d+)\.self_attn\.([qkvo])_proj\.weight\Z"
)


def prune_projection(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Prune the requested row or column block from one attention weight."""
    match = ATTENTION_KEY.fullmatch(name)
    if match is None:
        return tensor

    layer = int(match.group(1))
    projection = match.group(2)
    assert 0 <= layer < LAYERS, f"unexpected attention layer in {name}"
    assert tuple(tensor.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
        f"unexpected source shape for {name}: {tuple(tensor.shape)}"
    )

    start = REMOVED_HEAD * HEAD_DIM
    stop = start + HEAD_DIM
    axis = 1 if projection == "o" else 0
    before = tensor.narrow(axis, 0, start)
    after = tensor.narrow(axis, stop, HIDDEN_SIZE - stop)
    result = torch.cat((before, after), dim=axis)

    expected = (
        (HIDDEN_SIZE, PRUNED_SIZE)
        if projection == "o"
        else (PRUNED_SIZE, HIDDEN_SIZE)
    )
    assert tuple(result.shape) == expected, (
        f"wrong pruned shape for {name}: {tuple(result.shape)} != {expected}"
    )
    assert result.dtype == tensor.dtype, f"dtype changed for {name}"
    return result


def main() -> None:
    with INDEX.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    assert len(weight_map) == EXPECTED_TENSORS, (
        f"input index has {len(weight_map)} tensors, expected {EXPECTED_TENSORS}"
    )

    # Load each shard once. safetensors keeps unmodified tensors mmap-backed;
    # only the pruned projections allocate new storage.
    shards: dict[str, dict[str, torch.Tensor]] = {}
    for shard_name in sorted(set(weight_map.values())):
        shards[shard_name] = load_file(BASE / shard_name, device="cpu")

    actual_keys = {key for shard in shards.values() for key in shard}
    expected_keys = set(weight_map)
    assert actual_keys == expected_keys, (
        f"shard/index key mismatch: missing={sorted(expected_keys - actual_keys)}, "
        f"extra={sorted(actual_keys - expected_keys)}"
    )

    output: dict[str, torch.Tensor] = {}
    pruned_names: set[str] = set()
    for name, shard_name in weight_map.items():
        source = shards[shard_name][name]
        result = prune_projection(name, source)
        output[name] = result
        if result is not source:
            pruned_names.add(name)

    expected_pruned = {
        f"model.layers.{layer}.self_attn.{projection}_proj.weight"
        for layer in range(LAYERS)
        for projection in ("q", "k", "v", "o")
    }
    assert pruned_names == expected_pruned, (
        f"wrong projection set: missing={sorted(expected_pruned - pruned_names)}, "
        f"extra={sorted(pruned_names - expected_pruned)}"
    )

    # Required pre-write checks from TASK.md.
    required_shapes = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for name, expected_shape in required_shapes.items():
        assert tuple(output[name].shape) == expected_shape, (
            f"required check failed for {name}: {tuple(output[name].shape)}"
        )
    assert len(output) == EXPECTED_TENSORS, (
        f"required tensor-count check failed: {len(output)}"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT, metadata={"format": "pt"})
    print(f"wrote {len(output)} tensors to {OUTPUT}")


if __name__ == "__main__":
    main()
