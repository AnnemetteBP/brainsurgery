from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 16


def prune_head_five(tensors: dict[str, torch.Tensor]) -> None:
    for layer in range(LAYERS):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight_key = f"{prefix}.query_key_value.weight"
        qkv_bias_key = f"{prefix}.query_key_value.bias"
        dense_weight_key = f"{prefix}.dense.weight"

        qkv_weight = tensors[qkv_weight_key]
        qkv_bias = tensors[qkv_bias_key]
        dense_weight = tensors[dense_weight_key]

        assert qkv_weight.shape == (6144, 2048), (
            qkv_weight_key,
            qkv_weight.shape,
        )
        assert qkv_bias.shape == (6144,), (qkv_bias_key, qkv_bias.shape)
        assert dense_weight.shape == (2048, 2048), (
            dense_weight_key,
            dense_weight.shape,
        )

        # Q, K, and V are grouped inside each 768-row head block.
        tensors[qkv_weight_key] = torch.cat(
            (qkv_weight[:3840], qkv_weight[4608:]), dim=0
        )
        tensors[qkv_bias_key] = torch.cat(
            (qkv_bias[:3840], qkv_bias[4608:]), dim=0
        )
        tensors[dense_weight_key] = torch.cat(
            (dense_weight[:, :1280], dense_weight[:, 1536:]), dim=1
        )

        assert tensors[qkv_weight_key].shape == (5376, 2048)
        assert tensors[qkv_bias_key].shape == (5376,)
        assert tensors[dense_weight_key].shape == (2048, 1792)


def main() -> None:
    tensors = load_file(INPUT, device="cpu")
    assert len(tensors) == 244, f"expected 244 input tensors, got {len(tensors)}"

    prune_head_five(tensors)

    # Required pre-write checks.
    layer_zero = "gpt_neox.layers.0.attention"
    assert tensors[f"{layer_zero}.query_key_value.weight"].shape == (5376, 2048)
    assert tensors[f"{layer_zero}.query_key_value.bias"].shape == (5376,)
    assert tensors[f"{layer_zero}.dense.weight"].shape == (2048, 1792)
    assert len(tensors) == 244, f"expected 244 output tensors, got {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
