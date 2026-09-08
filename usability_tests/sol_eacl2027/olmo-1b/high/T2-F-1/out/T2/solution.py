#!/usr/bin/env python3
"""Remove attention head 5 from every OLMo layer checkpoint tensor."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 16
HIDDEN_SIZE = 2048
HEAD_DIM = 128
HEAD_TO_REMOVE = 5
EXPECTED_TENSORS = 114


def main() -> None:
    with (BASE / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle)["weight_map"]

    assert len(weight_map) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} indexed tensors, found {len(weight_map)}"
    )

    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard = load_file(BASE / shard_name, device="cpu")
        overlap = tensors.keys() & shard.keys()
        assert not overlap, f"duplicate tensor names across shards: {sorted(overlap)}"
        tensors.update(shard)

    assert set(tensors) == set(weight_map), "loaded tensor keys differ from the index"

    start = HEAD_TO_REMOVE * HEAD_DIM
    end = start + HEAD_DIM
    modified: set[str] = set()

    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.{projection}.weight"
            original = tensors[name]
            assert tuple(original.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
                f"unexpected input shape for {name}: {tuple(original.shape)}"
            )
            tensors[name] = torch.cat((original[:start, :], original[end:, :]), dim=0)
            modified.add(name)

        name = f"{prefix}.o_proj.weight"
        original = tensors[name]
        assert tuple(original.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
            f"unexpected input shape for {name}: {tuple(original.shape)}"
        )
        tensors[name] = torch.cat((original[:, :start], original[:, end:]), dim=1)
        modified.add(name)

    expected_modified = {
        f"model.layers.{layer}.self_attn.{projection}.weight"
        for layer in range(LAYERS)
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    assert modified == expected_modified, "the set of modified tensors is incorrect"

    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.{projection}.weight"
            assert tuple(tensors[name].shape) == (1920, 2048), (
                f"incorrect output shape for {name}: {tuple(tensors[name].shape)}"
            )
        name = f"{prefix}.o_proj.weight"
        assert tuple(tensors[name].shape) == (2048, 1920), (
            f"incorrect output shape for {name}: {tuple(tensors[name].shape)}"
        )

    # Required pre-write checks, kept explicit so any violation aborts the run.
    assert tuple(tensors["model.layers.0.self_attn.q_proj.weight"].shape) == (1920, 2048)
    assert tuple(tensors["model.layers.0.self_attn.k_proj.weight"].shape) == (1920, 2048)
    assert tuple(tensors["model.layers.0.self_attn.v_proj.weight"].shape) == (1920, 2048)
    assert tuple(tensors["model.layers.0.self_attn.o_proj.weight"].shape) == (2048, 1920)
    assert len(tensors) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} output tensors, found {len(tensors)}"
    )

    save_file(tensors, OUTPUT)
    print(f"wrote {len(tensors)} tensors to {OUTPUT}")


if __name__ == "__main__":
    main()
