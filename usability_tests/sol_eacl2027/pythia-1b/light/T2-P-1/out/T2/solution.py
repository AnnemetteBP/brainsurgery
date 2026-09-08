from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
NUM_LAYERS = 16


def main() -> None:
    tensors = load_file(INPUT)

    for layer in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight = f"{prefix}.query_key_value.weight"
        qkv_bias = f"{prefix}.query_key_value.bias"
        dense_weight = f"{prefix}.dense.weight"

        assert tensors[qkv_weight].shape == torch.Size([6144, 2048]), qkv_weight
        assert tensors[qkv_bias].shape == torch.Size([6144]), qkv_bias
        assert tensors[dense_weight].shape == torch.Size([2048, 2048]), dense_weight

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

        assert tensors[qkv_weight].shape == torch.Size([5376, 2048]), qkv_weight
        assert tensors[qkv_bias].shape == torch.Size([5376]), qkv_bias
        assert tensors[dense_weight].shape == torch.Size([2048, 1792]), dense_weight

    assert tensors["gpt_neox.layers.0.attention.query_key_value.weight"].shape == torch.Size([5376, 2048])
    assert tensors["gpt_neox.layers.0.attention.query_key_value.bias"].shape == torch.Size([5376])
    assert tensors["gpt_neox.layers.0.attention.dense.weight"].shape == torch.Size([2048, 1792])
    assert len(tensors) == 244, f"expected 244 tensors, found {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
