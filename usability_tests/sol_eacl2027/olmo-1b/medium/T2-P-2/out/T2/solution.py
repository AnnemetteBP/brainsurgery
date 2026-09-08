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
HEAD_DIM = 128
HEAD_TO_REMOVE = 5
EXPECTED_TENSORS = 114


def load_sharded_checkpoint() -> dict[str, torch.Tensor]:
    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    if len(weight_map) != EXPECTED_TENSORS:
        raise RuntimeError(
            f"Expected {EXPECTED_TENSORS} indexed tensors, found {len(weight_map)}"
        )

    tensors: dict[str, torch.Tensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = BASE_DIR / shard_name
        expected_keys = {key for key, shard in weight_map.items() if shard == shard_name}
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            actual_keys = set(shard.keys())
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise RuntimeError(
                    f"Index mismatch for {shard_name}: missing={missing}, extra={extra}"
                )
            for key in shard.keys():
                tensors[key] = shard.get_tensor(key)

    if set(tensors) != set(weight_map):
        raise RuntimeError("Loaded checkpoint keys do not match the index")
    return tensors


def prune_attention_heads(tensors: dict[str, torch.Tensor]) -> None:
    start = HEAD_TO_REMOVE * HEAD_DIM
    stop = start + HEAD_DIM
    original_shape = (HIDDEN_SIZE, HIDDEN_SIZE)

    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}.self_attn"
        for projection in ("q_proj", "k_proj", "v_proj"):
            key = f"{prefix}.{projection}.weight"
            tensor = tensors.get(key)
            if tensor is None:
                raise RuntimeError(f"Missing required tensor: {key}")
            if tuple(tensor.shape) != original_shape:
                raise RuntimeError(
                    f"Unexpected source shape for {key}: {tuple(tensor.shape)}"
                )
            tensors[key] = torch.cat((tensor[:start, :], tensor[stop:, :]), dim=0)

        output_key = f"{prefix}.o_proj.weight"
        tensor = tensors.get(output_key)
        if tensor is None:
            raise RuntimeError(f"Missing required tensor: {output_key}")
        if tuple(tensor.shape) != original_shape:
            raise RuntimeError(
                f"Unexpected source shape for {output_key}: {tuple(tensor.shape)}"
            )
        tensors[output_key] = torch.cat(
            (tensor[:, :start], tensor[:, stop:]), dim=1
        )


def validate_required_result(tensors: dict[str, torch.Tensor]) -> None:
    required_shapes = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for key, expected_shape in required_shapes.items():
        actual_shape = tuple(tensors[key].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Required check failed for {key}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    if len(tensors) != EXPECTED_TENSORS:
        raise RuntimeError(
            f"Required check failed: expected {EXPECTED_TENSORS} tensors, "
            f"got {len(tensors)}"
        )


def main() -> None:
    tensors = load_sharded_checkpoint()
    prune_attention_heads(tensors)
    validate_required_result(tensors)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT_PATH)
    print(f"Wrote {len(tensors)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
