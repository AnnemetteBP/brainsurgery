from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T2/model.safetensors")
NUM_LAYERS = 12


def require_shape(tensors: dict[str, torch.Tensor], name: str, shape: tuple[int, ...]) -> None:
    if name not in tensors:
        raise KeyError(f"required tensor is missing: {name}")
    actual = tuple(tensors[name].shape)
    if actual != shape:
        raise ValueError(f"{name} has shape {actual}, expected {shape}")


def main() -> None:
    tensors = load_file(INPUT_PATH)
    if len(tensors) != 160:
        raise ValueError(f"input has {len(tensors)} tensors, expected 160")

    qkv_ranges = (
        (0, 320),
        (384, 768),
        (768, 1088),
        (1152, 1536),
        (1536, 1856),
        (1920, 2304),
    )
    projection_ranges = ((0, 320), (384, 768))

    for layer in range(NUM_LAYERS):
        prefix = f"h.{layer}.attn"
        qkv_weight = f"{prefix}.c_attn.weight"
        qkv_bias = f"{prefix}.c_attn.bias"
        projection_weight = f"{prefix}.c_proj.weight"

        require_shape(tensors, qkv_weight, (768, 2304))
        require_shape(tensors, qkv_bias, (2304,))
        require_shape(tensors, projection_weight, (768, 768))

        tensors[qkv_weight] = torch.cat(
            [tensors[qkv_weight][:, start:end] for start, end in qkv_ranges], dim=1
        )
        tensors[qkv_bias] = torch.cat(
            [tensors[qkv_bias][start:end] for start, end in qkv_ranges], dim=0
        )
        tensors[projection_weight] = torch.cat(
            [tensors[projection_weight][start:end, :] for start, end in projection_ranges],
            dim=0,
        )

        require_shape(tensors, qkv_weight, (768, 2112))
        require_shape(tensors, qkv_bias, (2112,))
        require_shape(tensors, projection_weight, (704, 768))

    # Required pre-write checks from the task specification.
    require_shape(tensors, "h.0.attn.c_attn.weight", (768, 2112))
    require_shape(tensors, "h.0.attn.c_attn.bias", (2112,))
    require_shape(tensors, "h.0.attn.c_proj.weight", (704, 768))
    if len(tensors) != 160:
        raise ValueError(f"output would have {len(tensors)} tensors, expected 160")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT_PATH)


if __name__ == "__main__":
    main()
