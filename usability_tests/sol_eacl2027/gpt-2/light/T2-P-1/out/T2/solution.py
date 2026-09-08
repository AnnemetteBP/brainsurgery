from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 12
HIDDEN = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5


def without_head(tensor: torch.Tensor, axis: int, segment_start: int = 0) -> torch.Tensor:
    """Remove one 64-wide head block from an indicated 768-wide segment."""
    cut_start = segment_start + HEAD_TO_REMOVE * HEAD_SIZE
    cut_end = cut_start + HEAD_SIZE
    before = tensor.narrow(axis, segment_start, cut_start - segment_start)
    after = tensor.narrow(axis, cut_end, segment_start + HIDDEN - cut_end)
    return torch.cat((before, after), dim=axis)


def main() -> None:
    tensors = load_file(INPUT)

    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        weight_key = f"{prefix}.c_attn.weight"
        bias_key = f"{prefix}.c_attn.bias"
        proj_key = f"{prefix}.c_proj.weight"

        weight = tensors[weight_key]
        bias = tensors[bias_key]
        proj = tensors[proj_key]
        assert tuple(weight.shape) == (768, 2304), (weight_key, weight.shape)
        assert tuple(bias.shape) == (2304,), (bias_key, bias.shape)
        assert tuple(proj.shape) == (768, 768), (proj_key, proj.shape)

        # Prune the same head independently from the fused q, k, and v segments.
        tensors[weight_key] = torch.cat(
            [without_head(weight[:, start : start + HIDDEN], 1) for start in range(0, 3 * HIDDEN, HIDDEN)],
            dim=1,
        )
        tensors[bias_key] = torch.cat(
            [without_head(bias[start : start + HIDDEN], 0) for start in range(0, 3 * HIDDEN, HIDDEN)],
            dim=0,
        )
        tensors[proj_key] = without_head(proj, 0)

        assert tuple(tensors[weight_key].shape) == (768, 2112), weight_key
        assert tuple(tensors[bias_key].shape) == (2112,), bias_key
        assert tuple(tensors[proj_key].shape) == (704, 768), proj_key

    # Required pre-write checks.
    assert tuple(tensors["h.0.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(tensors["h.0.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(tensors["h.0.attn.c_proj.weight"].shape) == (704, 768)
    assert len(tensors) == 160, f"expected 160 tensors, found {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
