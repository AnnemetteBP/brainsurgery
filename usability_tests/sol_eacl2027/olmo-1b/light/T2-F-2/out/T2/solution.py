#!/usr/bin/env python3
"""Remove attention head 5 from every OLMo layer."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 16
HIDDEN = 2048
HEAD_START = 5 * 128
HEAD_END = 6 * 128


def main() -> None:
    with (BASE / "model.safetensors.index.json").open() as handle:
        index = json.load(handle)

    weight_map = index["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard = load_file(BASE / shard_name, device="cpu")
        overlap = tensors.keys() & shard.keys()
        assert not overlap, f"duplicate tensor keys across shards: {sorted(overlap)}"
        tensors.update(shard)

    assert set(tensors) == set(weight_map), "loaded tensor keys differ from index"
    assert len(tensors) == 114, f"expected 114 input tensors, got {len(tensors)}"

    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            key = f"{prefix}.{projection}.weight"
            tensor = tensors[key]
            assert tuple(tensor.shape) == (HIDDEN, HIDDEN), (
                f"unexpected input shape for {key}: {tuple(tensor.shape)}"
            )
            tensors[key] = torch.cat(
                (tensor[:HEAD_START, :], tensor[HEAD_END:, :]), dim=0
            )

        key = f"{prefix}.o_proj.weight"
        tensor = tensors[key]
        assert tuple(tensor.shape) == (HIDDEN, HIDDEN), (
            f"unexpected input shape for {key}: {tuple(tensor.shape)}"
        )
        tensors[key] = torch.cat(
            (tensor[:, :HEAD_START], tensor[:, HEAD_END:]), dim=1
        )

    # Required checks are deliberately performed before creating the output.
    layer0 = "model.layers.0.self_attn"
    assert tuple(tensors[f"{layer0}.q_proj.weight"].shape) == (1920, 2048)
    assert tuple(tensors[f"{layer0}.k_proj.weight"].shape) == (1920, 2048)
    assert tuple(tensors[f"{layer0}.v_proj.weight"].shape) == (1920, 2048)
    assert tuple(tensors[f"{layer0}.o_proj.weight"].shape) == (2048, 1920)
    assert len(tensors) == 114, f"expected 114 output tensors, got {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
