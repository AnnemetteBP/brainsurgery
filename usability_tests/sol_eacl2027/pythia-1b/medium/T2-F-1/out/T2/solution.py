"""Remove attention head 5 from every Pythia-1B layer."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 16
EXPECTED_TENSORS = 244


def require_shape(tensors: dict[str, torch.Tensor], name: str, shape: tuple[int, ...]) -> None:
    actual = tuple(tensors[name].shape)
    if actual != shape:
        raise AssertionError(f"{name}: expected shape {shape}, got {actual}")


def main() -> None:
    tensors = load_file(INPUT, device="cpu")
    if len(tensors) != EXPECTED_TENSORS:
        raise AssertionError(
            f"input: expected {EXPECTED_TENSORS} tensors, got {len(tensors)}"
        )

    for layer in range(LAYERS):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight = f"{prefix}.query_key_value.weight"
        qkv_bias = f"{prefix}.query_key_value.bias"
        dense_weight = f"{prefix}.dense.weight"

        require_shape(tensors, qkv_weight, (6144, 2048))
        require_shape(tensors, qkv_bias, (6144,))
        require_shape(tensors, dense_weight, (2048, 2048))

        # Head 5 occupies QKV rows 5*768:6*768 and dense columns
        # 5*256:6*256. Concatenation preserves the order of all other heads.
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

    # Required checks are deliberately performed before save_file.
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
    if len(tensors) != EXPECTED_TENSORS:
        raise AssertionError(
            f"output: expected {EXPECTED_TENSORS} tensors, got {len(tensors)}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
