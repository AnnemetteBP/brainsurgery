from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T2/model.safetensors")
NUM_LAYERS = 16
QKV_HEAD_BLOCK = 3 * 256
HEAD_TO_REMOVE = 5
HEAD_WIDTH = 256


def require_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise RuntimeError(f"{name} has shape {actual}, expected {expected}")


def main() -> None:
    tensors = load_file(INPUT_PATH)
    if len(tensors) != 244:
        raise RuntimeError(f"input has {len(tensors)} tensors, expected 244")

    output = dict(tensors)
    qkv_start = HEAD_TO_REMOVE * QKV_HEAD_BLOCK
    qkv_end = qkv_start + QKV_HEAD_BLOCK
    dense_start = HEAD_TO_REMOVE * HEAD_WIDTH
    dense_end = dense_start + HEAD_WIDTH

    for layer in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight_name = f"{prefix}.query_key_value.weight"
        qkv_bias_name = f"{prefix}.query_key_value.bias"
        dense_weight_name = f"{prefix}.dense.weight"

        qkv_weight = tensors[qkv_weight_name]
        qkv_bias = tensors[qkv_bias_name]
        dense_weight = tensors[dense_weight_name]

        require_shape(qkv_weight, (6144, 2048), qkv_weight_name)
        require_shape(qkv_bias, (6144,), qkv_bias_name)
        require_shape(dense_weight, (2048, 2048), dense_weight_name)

        output[qkv_weight_name] = torch.cat(
            (qkv_weight[:qkv_start], qkv_weight[qkv_end:]), dim=0
        )
        output[qkv_bias_name] = torch.cat(
            (qkv_bias[:qkv_start], qkv_bias[qkv_end:]), dim=0
        )
        output[dense_weight_name] = torch.cat(
            (dense_weight[:, :dense_start], dense_weight[:, dense_end:]), dim=1
        )

        require_shape(output[qkv_weight_name], (5376, 2048), qkv_weight_name)
        require_shape(output[qkv_bias_name], (5376,), qkv_bias_name)
        require_shape(output[dense_weight_name], (2048, 1792), dense_weight_name)

    # Explicit required checks, performed before writing the checkpoint.
    require_shape(
        output["gpt_neox.layers.0.attention.query_key_value.weight"],
        (5376, 2048),
        "gpt_neox.layers.0.attention.query_key_value.weight",
    )
    require_shape(
        output["gpt_neox.layers.0.attention.query_key_value.bias"],
        (5376,),
        "gpt_neox.layers.0.attention.query_key_value.bias",
    )
    require_shape(
        output["gpt_neox.layers.0.attention.dense.weight"],
        (2048, 1792),
        "gpt_neox.layers.0.attention.dense.weight",
    )
    if len(output) != 244:
        raise RuntimeError(f"output has {len(output)} tensors, expected 244")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT_PATH)


if __name__ == "__main__":
    main()
