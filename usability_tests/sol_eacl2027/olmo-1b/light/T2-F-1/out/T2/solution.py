#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T2/model.safetensors")
INDEX = BASE / "model.safetensors.index.json"
LAYERS = 16
ORIGINAL_SHAPE = (2048, 2048)
HEAD_START = 5 * 128
HEAD_END = 6 * 128


def main() -> None:
    with INDEX.open() as handle:
        weight_map = json.load(handle)["weight_map"]

    assert len(weight_map) == 114, f"expected 114 indexed tensors, got {len(weight_map)}"
    tensors = {}
    for shard_name in sorted(set(weight_map.values())):
        shard = load_file(BASE / shard_name, device="cpu")
        overlap = tensors.keys() & shard.keys()
        assert not overlap, f"duplicate tensors across shards: {sorted(overlap)}"
        tensors.update(shard)
    assert set(tensors) == set(weight_map), "loaded keys differ from index keys"

    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            name = f"{prefix}.{projection}.weight"
            tensor = tensors[name]
            assert tuple(tensor.shape) == ORIGINAL_SHAPE, (
                f"{name}: expected source shape {ORIGINAL_SHAPE}, got {tuple(tensor.shape)}"
            )
            if projection == "o_proj":
                tensors[name] = torch.cat(
                    (tensor[:, :HEAD_START], tensor[:, HEAD_END:]), dim=1
                )
            else:
                tensors[name] = torch.cat(
                    (tensor[:HEAD_START, :], tensor[HEAD_END:, :]), dim=0
                )

    # Required pre-write checks.
    expected = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for name, shape in expected.items():
        assert tuple(tensors[name].shape) == shape, (
            f"{name}: expected output shape {shape}, got {tuple(tensors[name].shape)}"
        )
    assert len(tensors) == 114, f"expected 114 output tensors, got {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)
    print(f"wrote {OUTPUT} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
