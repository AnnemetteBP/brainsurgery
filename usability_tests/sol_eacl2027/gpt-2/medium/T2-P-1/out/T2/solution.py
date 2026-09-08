from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T2/model.safetensors")
NUM_LAYERS = 12
HIDDEN_SIZE = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5


def keep_without_head(tensor: torch.Tensor, axis: int, segment_starts: tuple[int, ...]) -> torch.Tensor:
    """Remove one 64-element head block from each indicated 768-wide segment."""
    pieces = []
    cut_start = HEAD_TO_REMOVE * HEAD_SIZE
    cut_end = cut_start + HEAD_SIZE

    for segment_start in segment_starts:
        before = tensor.narrow(axis, segment_start, cut_start)
        after = tensor.narrow(
            axis,
            segment_start + cut_end,
            HIDDEN_SIZE - cut_end,
        )
        pieces.extend((before, after))

    return torch.cat(pieces, dim=axis).contiguous()


def main() -> None:
    tensors = load_file(INPUT_PATH)

    for layer in range(NUM_LAYERS):
        prefix = f"h.{layer}.attn"
        c_attn_weight = f"{prefix}.c_attn.weight"
        c_attn_bias = f"{prefix}.c_attn.bias"
        c_proj_weight = f"{prefix}.c_proj.weight"

        assert tuple(tensors[c_attn_weight].shape) == (768, 2304), (
            c_attn_weight,
            tuple(tensors[c_attn_weight].shape),
        )
        assert tuple(tensors[c_attn_bias].shape) == (2304,), (
            c_attn_bias,
            tuple(tensors[c_attn_bias].shape),
        )
        assert tuple(tensors[c_proj_weight].shape) == (768, 768), (
            c_proj_weight,
            tuple(tensors[c_proj_weight].shape),
        )

        tensors[c_attn_weight] = keep_without_head(
            tensors[c_attn_weight], axis=1, segment_starts=(0, 768, 1536)
        )
        tensors[c_attn_bias] = keep_without_head(
            tensors[c_attn_bias], axis=0, segment_starts=(0, 768, 1536)
        )
        tensors[c_proj_weight] = keep_without_head(
            tensors[c_proj_weight], axis=0, segment_starts=(0,)
        )

        assert tuple(tensors[c_attn_weight].shape) == (768, 2112)
        assert tuple(tensors[c_attn_bias].shape) == (2112,)
        assert tuple(tensors[c_proj_weight].shape) == (704, 768)

    # Required pre-write checks.
    assert tuple(tensors["h.0.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(tensors["h.0.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(tensors["h.0.attn.c_proj.weight"].shape) == (704, 768)
    assert len(tensors) == 160, f"expected 160 tensors, found {len(tensors)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT_PATH)


if __name__ == "__main__":
    main()
