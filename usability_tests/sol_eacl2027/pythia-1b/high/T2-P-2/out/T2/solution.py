from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T2/model.safetensors")
NUM_LAYERS = 16

QKV_INPUT_SHAPE = (6144, 2048)
QKV_BIAS_INPUT_SHAPE = (6144,)
DENSE_INPUT_SHAPE = (2048, 2048)

QKV_OUTPUT_SHAPE = (5376, 2048)
QKV_BIAS_OUTPUT_SHAPE = (5376,)
DENSE_OUTPUT_SHAPE = (2048, 1792)


def require_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise RuntimeError(f"{name} has shape {actual}, expected {expected}")


def main() -> None:
    tensors = load_file(INPUT_PATH, device="cpu")
    if len(tensors) != 244:
        raise RuntimeError(f"input has {len(tensors)} tensors, expected 244")

    for layer in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight_name = f"{prefix}.query_key_value.weight"
        qkv_bias_name = f"{prefix}.query_key_value.bias"
        dense_weight_name = f"{prefix}.dense.weight"

        for name in (qkv_weight_name, qkv_bias_name, dense_weight_name):
            if name not in tensors:
                raise RuntimeError(f"missing required tensor: {name}")

        qkv_weight = tensors[qkv_weight_name]
        qkv_bias = tensors[qkv_bias_name]
        dense_weight = tensors[dense_weight_name]
        require_shape(qkv_weight, QKV_INPUT_SHAPE, qkv_weight_name)
        require_shape(qkv_bias, QKV_BIAS_INPUT_SHAPE, qkv_bias_name)
        require_shape(dense_weight, DENSE_INPUT_SHAPE, dense_weight_name)

        # Head 5 is rows 5*768:6*768 in interleaved Q/K/V storage.
        tensors[qkv_weight_name] = torch.cat(
            (qkv_weight[:3840], qkv_weight[4608:]), dim=0
        )
        tensors[qkv_bias_name] = torch.cat(
            (qkv_bias[:3840], qkv_bias[4608:]), dim=0
        )

        # The same head is columns 5*256:6*256 in the output projection.
        tensors[dense_weight_name] = torch.cat(
            (dense_weight[:, :1280], dense_weight[:, 1536:]), dim=1
        )

        require_shape(tensors[qkv_weight_name], QKV_OUTPUT_SHAPE, qkv_weight_name)
        require_shape(tensors[qkv_bias_name], QKV_BIAS_OUTPUT_SHAPE, qkv_bias_name)
        require_shape(tensors[dense_weight_name], DENSE_OUTPUT_SHAPE, dense_weight_name)

    # Required pre-write checks from the task.
    require_shape(
        tensors["gpt_neox.layers.0.attention.query_key_value.weight"],
        QKV_OUTPUT_SHAPE,
        "gpt_neox.layers.0.attention.query_key_value.weight",
    )
    require_shape(
        tensors["gpt_neox.layers.0.attention.query_key_value.bias"],
        QKV_BIAS_OUTPUT_SHAPE,
        "gpt_neox.layers.0.attention.query_key_value.bias",
    )
    require_shape(
        tensors["gpt_neox.layers.0.attention.dense.weight"],
        DENSE_OUTPUT_SHAPE,
        "gpt_neox.layers.0.attention.dense.weight",
    )
    if len(tensors) != 244:
        raise RuntimeError(f"output has {len(tensors)} tensors, expected 244")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT_PATH)


if __name__ == "__main__":
    main()
