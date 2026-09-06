"""
Structured attention-head pruning for GPT-2 (124M).

Removes head 5 (0-indexed) from every layer's attention block:
- attn.c_attn.weight: drop the 64-wide column slice for head 5 within each
  of the three 768-wide [q|k|v] segments.
- attn.c_attn.bias: same column slicing, applied to the bias vector.
- attn.c_proj.weight: drop the 64-wide row slice for head 5 (output proj
  consumes heads as rows).

All other tensors are copied through unchanged.
"""

import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T2/model.safetensors")

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM  # 768
HEAD_TO_PRUNE = 5

assert HIDDEN == 768


def keep_columns_mask(width: int, head_dim: int, head_to_prune: int) -> torch.Tensor:
    """Boolean mask over `width` columns/rows, dropping the slice for one head."""
    num_heads = width // head_dim
    assert num_heads * head_dim == width
    mask = torch.ones(width, dtype=torch.bool)
    start = head_to_prune * head_dim
    mask[start : start + head_dim] = False
    return mask


def prune_c_attn_axis_mask() -> torch.Tensor:
    """Mask over the 2304-wide [q|k|v] axis: drop head 5's 64 columns in each segment."""
    seg_mask = keep_columns_mask(HIDDEN, HEAD_DIM, HEAD_TO_PRUNE)  # 768 -> keep 704
    return torch.cat([seg_mask, seg_mask, seg_mask])  # 2304


def main() -> None:
    state_dict = load_file(str(INPUT_PATH))

    orig_count = len(state_dict)
    assert orig_count == 160, f"expected 160 input tensors, got {orig_count}"

    c_attn_mask = prune_c_attn_axis_mask()  # shape [2304], keeps 2112
    c_proj_row_mask = keep_columns_mask(HIDDEN, HEAD_DIM, HEAD_TO_PRUNE)  # shape [768], keeps 704

    c_attn_w_re = re.compile(r"^h\.(\d+)\.attn\.c_attn\.weight$")
    c_attn_b_re = re.compile(r"^h\.(\d+)\.attn\.c_attn\.bias$")
    c_proj_w_re = re.compile(r"^h\.(\d+)\.attn\.c_proj\.weight$")

    out: dict[str, torch.Tensor] = {}
    touched_layers = set()

    for name, tensor in state_dict.items():
        m = c_attn_w_re.match(name)
        if m:
            layer = int(m.group(1))
            assert tensor.shape == (HIDDEN, 3 * HIDDEN), f"{name} shape {tensor.shape}"
            out[name] = tensor[:, c_attn_mask].contiguous()
            touched_layers.add(layer)
            continue

        m = c_attn_b_re.match(name)
        if m:
            layer = int(m.group(1))
            assert tensor.shape == (3 * HIDDEN,), f"{name} shape {tensor.shape}"
            out[name] = tensor[c_attn_mask].contiguous()
            touched_layers.add(layer)
            continue

        m = c_proj_w_re.match(name)
        if m:
            layer = int(m.group(1))
            assert tensor.shape == (HIDDEN, HIDDEN), f"{name} shape {tensor.shape}"
            out[name] = tensor[c_proj_row_mask, :].contiguous()
            touched_layers.add(layer)
            continue

        # Untouched tensor: copy through as-is.
        out[name] = tensor

    assert touched_layers == set(range(NUM_LAYERS)), (
        f"expected all {NUM_LAYERS} layers touched, got {sorted(touched_layers)}"
    )

    # Required checks.
    expected_c_attn_w = (HIDDEN, 2112)
    expected_c_attn_b = (2112,)
    expected_c_proj_w = (704, HIDDEN)

    got = out["h.0.attn.c_attn.weight"].shape
    assert tuple(got) == expected_c_attn_w, f"h.0.attn.c_attn.weight shape {got}"

    got = out["h.0.attn.c_attn.bias"].shape
    assert tuple(got) == expected_c_attn_b, f"h.0.attn.c_attn.bias shape {got}"

    got = out["h.0.attn.c_proj.weight"].shape
    assert tuple(got) == expected_c_proj_w, f"h.0.attn.c_proj.weight shape {got}"

    for i in range(NUM_LAYERS):
        assert tuple(out[f"h.{i}.attn.c_attn.weight"].shape) == expected_c_attn_w
        assert tuple(out[f"h.{i}.attn.c_attn.bias"].shape) == expected_c_attn_b
        assert tuple(out[f"h.{i}.attn.c_proj.weight"].shape) == expected_c_proj_w
        # untouched
        assert tuple(out[f"h.{i}.attn.c_proj.bias"].shape) == (HIDDEN,)
        assert tuple(out[f"h.{i}.attn.bias"].shape) == (1, 1, 1024, 1024)

    assert len(out) == 160, f"expected 160 output tensors, got {len(out)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUTPUT_PATH))
    print(f"Wrote {len(out)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
