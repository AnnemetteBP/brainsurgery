#!/usr/bin/env python3
"""Remove attention head 5 from every OLMo layer and save one checkpoint."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("inputs/base")
INDEX = BASE / "model.safetensors.index.json"
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 16
HIDDEN_SIZE = 2048
HEAD_SIZE = 128
HEAD_TO_REMOVE = 5
EXPECTED_TENSORS = 114


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    assert len(weight_map) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} indexed tensors, found {len(weight_map)}"
    )

    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard = load_file(BASE / shard_name, device="cpu")
        overlap = tensors.keys() & shard.keys()
        assert not overlap, f"duplicate tensor names across shards: {sorted(overlap)}"
        tensors.update(shard)

    assert set(tensors) == set(weight_map), "loaded tensor keys differ from index"
    assert len(tensors) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} loaded tensors, found {len(tensors)}"
    )

    start = HEAD_TO_REMOVE * HEAD_SIZE
    stop = start + HEAD_SIZE
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.{projection}.weight"
            tensor = tensors[name]
            assert tuple(tensor.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
                f"unexpected input shape for {name}: {tuple(tensor.shape)}"
            )
            tensors[name] = torch.cat((tensor[:start, :], tensor[stop:, :]), dim=0)

        name = f"{prefix}.o_proj.weight"
        tensor = tensors[name]
        assert tuple(tensor.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
            f"unexpected input shape for {name}: {tuple(tensor.shape)}"
        )
        tensors[name] = torch.cat((tensor[:, :start], tensor[:, stop:]), dim=1)

    # Required pre-write checks.
    expected_layer_zero = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for name, expected_shape in expected_layer_zero.items():
        actual_shape = tuple(tensors[name].shape)
        assert actual_shape == expected_shape, (
            f"required shape check failed for {name}: "
            f"expected {expected_shape}, got {actual_shape}"
        )
    assert len(tensors) == EXPECTED_TENSORS, (
        f"required tensor-count check failed: expected {EXPECTED_TENSORS}, "
        f"got {len(tensors)}"
    )

    # Also enforce the target shape on every affected projection.
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.{projection}.weight"
            assert tuple(tensors[name].shape) == (1920, 2048), name
        name = f"{prefix}.o_proj.weight"
        assert tuple(tensors[name].shape) == (2048, 1920), name

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
