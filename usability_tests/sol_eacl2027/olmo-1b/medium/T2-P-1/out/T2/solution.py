import json
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
SLICE_START = HEAD_TO_REMOVE * HEAD_SIZE
SLICE_END = SLICE_START + HEAD_SIZE


def load_checkpoint() -> dict[str, torch.Tensor]:
    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    weight_map = index["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        with safe_open(BASE_DIR / shard_name, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name in tensors:
                    raise RuntimeError(f"duplicate tensor name: {name}")
                tensors[name] = shard.get_tensor(name)

    expected_names = set(weight_map)
    if set(tensors) != expected_names:
        missing = sorted(expected_names - set(tensors))
        extra = sorted(set(tensors) - expected_names)
        raise RuntimeError(f"checkpoint key mismatch: missing={missing}, extra={extra}")
    return tensors


def prune_head(tensors: dict[str, torch.Tensor]) -> None:
    expected_input_shape = (HIDDEN_SIZE, HIDDEN_SIZE)
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{prefix}.{projection}.weight"
            tensor = tensors[name]
            if tuple(tensor.shape) != expected_input_shape:
                raise RuntimeError(f"{name}: expected {expected_input_shape}, got {tuple(tensor.shape)}")
            tensors[name] = torch.cat(
                (tensor[:SLICE_START, :], tensor[SLICE_END:, :]), dim=0
            )

        name = f"{prefix}.o_proj.weight"
        tensor = tensors[name]
        if tuple(tensor.shape) != expected_input_shape:
            raise RuntimeError(f"{name}: expected {expected_input_shape}, got {tuple(tensor.shape)}")
        tensors[name] = torch.cat(
            (tensor[:, :SLICE_START], tensor[:, SLICE_END:]), dim=1
        )


def check_required_result(tensors: dict[str, torch.Tensor]) -> None:
    required_shapes = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for name, expected_shape in required_shapes.items():
        actual_shape = tuple(tensors[name].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(f"{name}: expected {expected_shape}, got {actual_shape}")

    if len(tensors) != 114:
        raise RuntimeError(f"expected exactly 114 tensors, got {len(tensors)}")


def main() -> None:
    tensors = load_checkpoint()
    prune_head(tensors)
    check_required_result(tensors)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT_PATH)
    print(f"Wrote {len(tensors)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
