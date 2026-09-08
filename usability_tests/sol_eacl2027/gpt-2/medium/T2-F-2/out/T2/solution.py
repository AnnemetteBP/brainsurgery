from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 12
HIDDEN_SIZE = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5


def without_head(tensor: torch.Tensor, axis: int, segment_starts: tuple[int, ...]) -> torch.Tensor:
    """Remove the selected head-sized slice from each indicated segment."""
    keep = torch.cat(
        [
            torch.arange(start, start + HIDDEN_SIZE)
            for start in segment_starts
        ]
    )
    remove_start = HEAD_TO_REMOVE * HEAD_SIZE
    within_segment = keep.remainder(HIDDEN_SIZE)
    keep = keep[(within_segment < remove_start) | (within_segment >= remove_start + HEAD_SIZE)]
    return tensor.index_select(axis, keep)


def main() -> None:
    tensors = load_file(INPUT)
    assert len(tensors) == 160, f"expected 160 input tensors, found {len(tensors)}"

    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        c_attn_weight = f"{prefix}.c_attn.weight"
        c_attn_bias = f"{prefix}.c_attn.bias"
        c_proj_weight = f"{prefix}.c_proj.weight"

        assert tuple(tensors[c_attn_weight].shape) == (768, 2304), c_attn_weight
        assert tuple(tensors[c_attn_bias].shape) == (2304,), c_attn_bias
        assert tuple(tensors[c_proj_weight].shape) == (768, 768), c_proj_weight

        tensors[c_attn_weight] = without_head(tensors[c_attn_weight], 1, (0, 768, 1536))
        tensors[c_attn_bias] = without_head(tensors[c_attn_bias], 0, (0, 768, 1536))
        tensors[c_proj_weight] = without_head(tensors[c_proj_weight], 0, (0,))

    # Required checks are deliberately performed before the output is written.
    assert tuple(tensors["h.0.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(tensors["h.0.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(tensors["h.0.attn.c_proj.weight"].shape) == (704, 768)
    assert len(tensors) == 160, f"expected 160 output tensors, found {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
