from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
NUM_LAYERS = 16


def require_shape(tensors, name, expected):
    actual = tuple(tensors[name].shape)
    if actual != expected:
        raise RuntimeError(f"{name}: expected shape {expected}, got {actual}")


def main():
    tensors = load_file(INPUT)

    for layer in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight = f"{prefix}.query_key_value.weight"
        qkv_bias = f"{prefix}.query_key_value.bias"
        dense_weight = f"{prefix}.dense.weight"

        require_shape(tensors, qkv_weight, (6144, 2048))
        require_shape(tensors, qkv_bias, (6144,))
        require_shape(tensors, dense_weight, (2048, 2048))

        tensors[qkv_weight] = torch.cat(
            (tensors[qkv_weight][:3840], tensors[qkv_weight][4608:]), dim=0
        )
        tensors[qkv_bias] = torch.cat(
            (tensors[qkv_bias][:3840], tensors[qkv_bias][4608:]), dim=0
        )
        tensors[dense_weight] = torch.cat(
            (tensors[dense_weight][:, :1280], tensors[dense_weight][:, 1536:]),
            dim=1,
        )

        require_shape(tensors, qkv_weight, (5376, 2048))
        require_shape(tensors, qkv_bias, (5376,))
        require_shape(tensors, dense_weight, (2048, 1792))

    # Required pre-write checks.
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
        tensors, "gpt_neox.layers.0.attention.dense.weight", (2048, 1792)
    )
    if len(tensors) != 244:
        raise RuntimeError(f"expected 244 tensors, got {len(tensors)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
