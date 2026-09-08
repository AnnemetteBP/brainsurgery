from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
NUM_LAYERS = 16
QKV_HEAD_WIDTH = 3 * 256
HEAD_WIDTH = 256
HEAD_TO_REMOVE = 5


def require_shape(tensors: dict[str, torch.Tensor], name: str, shape: tuple[int, ...]) -> None:
    if name not in tensors:
        raise KeyError(f"missing required tensor: {name}")
    actual = tuple(tensors[name].shape)
    if actual != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {actual}")


def main() -> None:
    tensors = load_file(INPUT, device="cpu")

    if len(tensors) != 244:
        raise ValueError(f"expected 244 input tensors, got {len(tensors)}")

    qkv_start = HEAD_TO_REMOVE * QKV_HEAD_WIDTH
    qkv_end = qkv_start + QKV_HEAD_WIDTH
    dense_start = HEAD_TO_REMOVE * HEAD_WIDTH
    dense_end = dense_start + HEAD_WIDTH

    for layer in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight = f"{prefix}.query_key_value.weight"
        qkv_bias = f"{prefix}.query_key_value.bias"
        dense_weight = f"{prefix}.dense.weight"

        require_shape(tensors, qkv_weight, (6144, 2048))
        require_shape(tensors, qkv_bias, (6144,))
        require_shape(tensors, dense_weight, (2048, 2048))

        tensors[qkv_weight] = torch.cat(
            (tensors[qkv_weight][:qkv_start], tensors[qkv_weight][qkv_end:]), dim=0
        )
        tensors[qkv_bias] = torch.cat(
            (tensors[qkv_bias][:qkv_start], tensors[qkv_bias][qkv_end:]), dim=0
        )
        tensors[dense_weight] = torch.cat(
            (tensors[dense_weight][:, :dense_start], tensors[dense_weight][:, dense_end:]),
            dim=1,
        )

        require_shape(tensors, qkv_weight, (5376, 2048))
        require_shape(tensors, qkv_bias, (5376,))
        require_shape(tensors, dense_weight, (2048, 1792))

    # Explicit task-mandated pre-write checks.
    require_shape(
        tensors,
        "gpt_neox.layers.0.attention.query_key_value.weight",
        (5376, 2048),
    )
    require_shape(
        tensors,
        "gpt_neox.layers.0.attention.query_key_value.bias",
        (5376,),
    )
    require_shape(
        tensors,
        "gpt_neox.layers.0.attention.dense.weight",
        (2048, 1792),
    )
    if len(tensors) != 244:
        raise ValueError(f"expected 244 output tensors, got {len(tensors)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
