import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T2/model.safetensors")
INDEX = BASE / "model.safetensors.index.json"
HEAD_START = 5 * 128
HEAD_END = 6 * 128


def main() -> None:
    with INDEX.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    tensors = {}
    for shard_name in sorted(set(weight_map.values())):
        tensors.update(load_file(BASE / shard_name, device="cpu"))

    assert len(tensors) == 114, f"expected 114 input tensors, got {len(tensors)}"
    assert set(tensors) == set(weight_map), "checkpoint keys differ from index"

    for layer in range(16):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.{projection}.weight"
            tensor = tensors[name]
            assert tuple(tensor.shape) == (2048, 2048), (
                f"unexpected input shape for {name}: {tuple(tensor.shape)}"
            )
            tensors[name] = torch.cat(
                (tensor[:HEAD_START, :], tensor[HEAD_END:, :]), dim=0
            ).contiguous()

        name = f"{prefix}.o_proj.weight"
        tensor = tensors[name]
        assert tuple(tensor.shape) == (2048, 2048), (
            f"unexpected input shape for {name}: {tuple(tensor.shape)}"
        )
        tensors[name] = torch.cat(
            (tensor[:, :HEAD_START], tensor[:, HEAD_END:]), dim=1
        ).contiguous()

    required_shapes = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for name, expected in required_shapes.items():
        assert tuple(tensors[name].shape) == expected, (
            f"wrong output shape for {name}: {tuple(tensors[name].shape)}"
        )
    assert len(tensors) == 114, f"expected 114 output tensors, got {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
