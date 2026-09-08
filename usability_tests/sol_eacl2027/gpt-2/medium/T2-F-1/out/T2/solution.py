from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 12
HIDDEN_SIZE = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5


def keep_without_head(segment_start: int) -> torch.Tensor:
    """Indices for one 768-wide Q, K, V, or projection-input segment."""
    removed_start = segment_start + HEAD_TO_REMOVE * HEAD_SIZE
    removed_end = removed_start + HEAD_SIZE
    return torch.cat(
        (
            torch.arange(segment_start, removed_start),
            torch.arange(removed_end, segment_start + HIDDEN_SIZE),
        )
    )


def main() -> None:
    tensors = load_file(INPUT)
    original_keys = set(tensors)

    qkv_keep = torch.cat(tuple(keep_without_head(start) for start in (0, 768, 1536)))
    projection_keep = keep_without_head(0)

    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        c_attn_weight = f"{prefix}.c_attn.weight"
        c_attn_bias = f"{prefix}.c_attn.bias"
        c_proj_weight = f"{prefix}.c_proj.weight"

        assert tensors[c_attn_weight].shape == (768, 2304), c_attn_weight
        assert tensors[c_attn_bias].shape == (2304,), c_attn_bias
        assert tensors[c_proj_weight].shape == (768, 768), c_proj_weight

        tensors[c_attn_weight] = tensors[c_attn_weight].index_select(1, qkv_keep)
        tensors[c_attn_bias] = tensors[c_attn_bias].index_select(0, qkv_keep)
        tensors[c_proj_weight] = tensors[c_proj_weight].index_select(0, projection_keep)

    # Required pre-write checks from the task.
    assert tensors["h.0.attn.c_attn.weight"].shape == (768, 2112)
    assert tensors["h.0.attn.c_attn.bias"].shape == (2112,)
    assert tensors["h.0.attn.c_proj.weight"].shape == (704, 768)
    assert len(tensors) == 160

    # Additional checks ensure every layer was transformed and no names changed.
    assert set(tensors) == original_keys
    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        assert tensors[f"{prefix}.c_attn.weight"].shape == (768, 2112)
        assert tensors[f"{prefix}.c_attn.bias"].shape == (2112,)
        assert tensors[f"{prefix}.c_proj.weight"].shape == (704, 768)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
