from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T2/model.safetensors")
NUM_LAYERS = 16


def require_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise AssertionError(f"{name}: expected shape {expected}, got {actual}")


def main() -> None:
    tensors = load_file(INPUT_PATH)
    output = dict(tensors)

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
            (qkv_weight[:3840], qkv_weight[4608:]), dim=0
        )
        output[qkv_bias_name] = torch.cat(
            (qkv_bias[:3840], qkv_bias[4608:]), dim=0
        )
        output[dense_weight_name] = torch.cat(
            (dense_weight[:, :1280], dense_weight[:, 1536:]), dim=1
        )

        require_shape(output[qkv_weight_name], (5376, 2048), qkv_weight_name)
        require_shape(output[qkv_bias_name], (5376,), qkv_bias_name)
        require_shape(output[dense_weight_name], (2048, 1792), dense_weight_name)

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
        raise AssertionError(f"expected exactly 244 tensors, got {len(output)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT_PATH)


if __name__ == "__main__":
    main()
