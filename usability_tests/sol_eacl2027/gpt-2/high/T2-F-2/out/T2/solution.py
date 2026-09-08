from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 12
HIDDEN_SIZE = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5


def keep_without_head(segment_width: int = HIDDEN_SIZE) -> torch.Tensor:
    """Indices for one Q/K/V or projection segment, excluding head 5."""
    start = HEAD_TO_REMOVE * HEAD_SIZE
    stop = start + HEAD_SIZE
    return torch.cat((torch.arange(start), torch.arange(stop, segment_width)))


def main() -> None:
    tensors = load_file(INPUT, device="cpu")
    assert len(tensors) == 160, f"expected 160 input tensors, got {len(tensors)}"

    per_segment = keep_without_head()
    qkv_columns = torch.cat(
        [per_segment + segment * HIDDEN_SIZE for segment in range(3)]
    )

    assert per_segment.numel() == 704
    assert qkv_columns.numel() == 2112

    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        c_attn_weight = f"{prefix}.c_attn.weight"
        c_attn_bias = f"{prefix}.c_attn.bias"
        c_proj_weight = f"{prefix}.c_proj.weight"

        assert tensors[c_attn_weight].shape == (768, 2304), (
            c_attn_weight,
            tensors[c_attn_weight].shape,
        )
        assert tensors[c_attn_bias].shape == (2304,), (
            c_attn_bias,
            tensors[c_attn_bias].shape,
        )
        assert tensors[c_proj_weight].shape == (768, 768), (
            c_proj_weight,
            tensors[c_proj_weight].shape,
        )

        # GPT-2 Conv1D weights are [in, out]. Q/K/V heads occupy columns
        # of c_attn, while heads consumed by c_proj occupy its rows.
        tensors[c_attn_weight] = tensors[c_attn_weight].index_select(1, qkv_columns)
        tensors[c_attn_bias] = tensors[c_attn_bias].index_select(0, qkv_columns)
        tensors[c_proj_weight] = tensors[c_proj_weight].index_select(0, per_segment)

        assert tensors[c_attn_weight].shape == (768, 2112)
        assert tensors[c_attn_bias].shape == (2112,)
        assert tensors[c_proj_weight].shape == (704, 768)

    # Required pre-write checks.
    assert tensors["h.0.attn.c_attn.weight"].shape == (768, 2112)
    assert tensors["h.0.attn.c_attn.bias"].shape == (2112,)
    assert tensors["h.0.attn.c_proj.weight"].shape == (704, 768)
    assert len(tensors) == 160, f"expected 160 output tensors, got {len(tensors)}"

    save_file(tensors, OUTPUT, metadata={"format": "pt"})


if __name__ == "__main__":
    main()
