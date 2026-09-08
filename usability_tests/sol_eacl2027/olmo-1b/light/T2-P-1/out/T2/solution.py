import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T2/model.safetensors")
INDEX = BASE / "model.safetensors.index.json"
LAYERS = 16
HIDDEN_SIZE = 2048
HEAD_DIM = 128
HEAD_TO_REMOVE = 5
EXPECTED_TENSORS = 114


def main() -> None:
    with INDEX.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    assert len(weight_map) == EXPECTED_TENSORS, (
        f"expected {EXPECTED_TENSORS} indexed tensors, found {len(weight_map)}"
    )

    tensors = {}
    for shard_name in sorted(set(weight_map.values())):
        expected_keys = {key for key, shard in weight_map.items() if shard == shard_name}
        with safe_open(BASE / shard_name, framework="pt", device="cpu") as shard:
            actual_keys = set(shard.keys())
            assert actual_keys == expected_keys, (
                f"index/key mismatch for {shard_name}: "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
            for key in shard.keys():
                tensors[key] = shard.get_tensor(key)

    assert set(tensors) == set(weight_map), "loaded tensor keys do not match the index"

    cut_start = HEAD_TO_REMOVE * HEAD_DIM
    cut_end = cut_start + HEAD_DIM
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            key = f"{prefix}.{projection}.weight"
            tensor = tensors[key]
            assert tuple(tensor.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
                f"unexpected input shape for {key}: {tuple(tensor.shape)}"
            )
            tensors[key] = torch.cat(
                (tensor[:cut_start, :], tensor[cut_end:, :]), dim=0
            )

        key = f"{prefix}.o_proj.weight"
        tensor = tensors[key]
        assert tuple(tensor.shape) == (HIDDEN_SIZE, HIDDEN_SIZE), (
            f"unexpected input shape for {key}: {tuple(tensor.shape)}"
        )
        tensors[key] = torch.cat(
            (tensor[:, :cut_start], tensor[:, cut_end:]), dim=1
        )

    required_shapes = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for key, expected_shape in required_shapes.items():
        assert tuple(tensors[key].shape) == expected_shape, (
            f"required check failed for {key}: got {tuple(tensors[key].shape)}, "
            f"expected {expected_shape}"
        )
    assert len(tensors) == EXPECTED_TENSORS, (
        f"required check failed: expected {EXPECTED_TENSORS} tensors, got {len(tensors)}"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)
    print(f"Wrote {len(tensors)} tensors to {OUTPUT}")


if __name__ == "__main__":
    main()
