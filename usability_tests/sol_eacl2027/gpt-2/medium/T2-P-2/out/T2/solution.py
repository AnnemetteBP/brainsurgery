from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T2/model.safetensors")
LAYERS = 12
HIDDEN_SIZE = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5


def without_head(tensor: torch.Tensor, axis: int, segment_starts: tuple[int, ...]) -> torch.Tensor:
    """Remove the selected head-sized block from each indicated segment."""
    pieces = []
    cut_start = HEAD_TO_REMOVE * HEAD_SIZE
    cut_end = cut_start + HEAD_SIZE
    for segment_start in segment_starts:
        before = tensor.narrow(axis, segment_start, cut_start)
        after = tensor.narrow(axis, segment_start + cut_end, HIDDEN_SIZE - cut_end)
        pieces.extend((before, after))
    return torch.cat(pieces, dim=axis).contiguous()


def main() -> None:
    tensors = load_file(INPUT_PATH)
    if len(tensors) != 160:
        raise ValueError(f"expected 160 input tensors, found {len(tensors)}")

    for layer in range(LAYERS):
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
                    f"{name} has shape {tuple(tensors[name].shape)}, expected {expected_shape}"
                )

        tensors[c_attn_weight] = without_head(
            tensors[c_attn_weight], axis=1, segment_starts=(0, 768, 1536)
        )
        tensors[c_attn_bias] = without_head(
            tensors[c_attn_bias], axis=0, segment_starts=(0, 768, 1536)
        )
        tensors[c_proj_weight] = without_head(
            tensors[c_proj_weight], axis=0, segment_starts=(0,)
        )

        expected_output_shapes = {
            c_attn_weight: (768, 2112),
            c_attn_bias: (2112,),
            c_proj_weight: (704, 768),
        }
        for name, expected_shape in expected_output_shapes.items():
            if tuple(tensors[name].shape) != expected_shape:
                raise AssertionError(
                    f"{name} has output shape {tuple(tensors[name].shape)}, expected {expected_shape}"
                )

    # Required pre-write checks, stated explicitly for layer 0 and tensor count.
    required_shapes = {
        "h.0.attn.c_attn.weight": (768, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, 768),
    }
    for name, expected_shape in required_shapes.items():
        if tuple(tensors[name].shape) != expected_shape:
            raise AssertionError(
                f"required check failed: {name} has shape {tuple(tensors[name].shape)}, "
                f"expected {expected_shape}"
            )
    if len(tensors) != 160:
        raise AssertionError(f"required check failed: expected 160 tensors, found {len(tensors)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT_PATH)


if __name__ == "__main__":
    main()
