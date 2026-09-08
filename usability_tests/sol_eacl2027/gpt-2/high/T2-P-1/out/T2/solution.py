from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T2/model.safetensors")
NUM_LAYERS = 12


def drop_head_from_qkv(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """Remove head 5 (columns/entries 320:384) from each Q, K, V segment."""
    pieces = []
    for segment_start in (0, 768, 1536):
        pieces.append(tensor.narrow(axis, segment_start, 320))
        pieces.append(tensor.narrow(axis, segment_start + 384, 384))
    return torch.cat(pieces, dim=axis)


def main() -> None:
    tensors = load_file(INPUT_PATH)

    if len(tensors) != 160:
        raise ValueError(f"expected 160 input tensors, found {len(tensors)}")

    output = dict(tensors)
    for layer in range(NUM_LAYERS):
        prefix = f"h.{layer}.attn"
        c_attn_weight = f"{prefix}.c_attn.weight"
        c_attn_bias = f"{prefix}.c_attn.bias"
        c_proj_weight = f"{prefix}.c_proj.weight"

        expected_input_shapes = {
            c_attn_weight: (768, 2304),
            c_attn_bias: (2304,),
            c_proj_weight: (768, 768),
        }
        for name, expected_shape in expected_input_shapes.items():
            if name not in tensors:
                raise KeyError(f"missing required tensor: {name}")
            if tuple(tensors[name].shape) != expected_shape:
                raise ValueError(
                    f"{name} has shape {tuple(tensors[name].shape)}, "
                    f"expected {expected_shape}"
                )

        output[c_attn_weight] = drop_head_from_qkv(tensors[c_attn_weight], axis=1)
        output[c_attn_bias] = drop_head_from_qkv(tensors[c_attn_bias], axis=0)
        output[c_proj_weight] = torch.cat(
            (tensors[c_proj_weight][:320, :], tensors[c_proj_weight][384:, :]),
            dim=0,
        )

        expected_output_shapes = {
            c_attn_weight: (768, 2112),
            c_attn_bias: (2112,),
            c_proj_weight: (704, 768),
        }
        for name, expected_shape in expected_output_shapes.items():
            if tuple(output[name].shape) != expected_shape:
                raise AssertionError(
                    f"{name} has output shape {tuple(output[name].shape)}, "
                    f"expected {expected_shape}"
                )

    # Required checks, performed before writing.
    required_shapes = {
        "h.0.attn.c_attn.weight": (768, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, 768),
    }
    for name, expected_shape in required_shapes.items():
        if tuple(output[name].shape) != expected_shape:
            raise AssertionError(
                f"{name} has output shape {tuple(output[name].shape)}, "
                f"expected {expected_shape}"
            )
    if len(output) != 160:
        raise AssertionError(f"expected 160 output tensors, found {len(output)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT_PATH)


if __name__ == "__main__":
    main()
