"""T2: structured attention-head pruning for GPT-2 (124M), condition F.

Removes head 5 (0-indexed) from every layer's attention block by slicing the
fused c_attn (q|k|v) weight/bias on the column axis and the c_proj weight on
the row axis, in each case dropping the 64-wide block that belongs to head 5
within every 768-wide segment (q, k, v for c_attn; the single 768-wide input
for c_proj).

No higher-level pruning API was used: the pinned transformers==5.12.1 GPT2
implementation (`transformers/models/gpt2/modeling_gpt2.py`) has no
`prune_heads` method on `GPT2Attention` or `GPT2Model` in this version (only
`transformers.pytorch_utils.prune_linear_layer` exists, which does not cover
Conv1D). So this is a direct safetensors + torch script, both allowed
packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T2/model.safetensors")


def keep_mask(n_heads: int, head_dim: int, pruned: set[int]) -> torch.Tensor:
    """Boolean mask over range(n_heads * head_dim), True for kept positions."""
    mask = torch.ones(n_heads * head_dim, dtype=torch.bool)
    for h in pruned:
        mask[h * head_dim : (h + 1) * head_dim] = False
    return mask


def main() -> None:
    tensors = load_file(IN_PATH)
    if len(tensors) != 160:
        sys.exit(f"expected 160 input tensors, got {len(tensors)}")

    head_mask = keep_mask(N_HEADS, HEAD_DIM, {PRUNE_HEAD})  # length 768, one head dropped
    kept_dim = int(head_mask.sum())  # 704

    out = dict(tensors)  # untouched tensors pass through unchanged

    for i in range(N_LAYERS):
        w_name = f"h.{i}.attn.c_attn.weight"
        b_name = f"h.{i}.attn.c_attn.bias"
        p_name = f"h.{i}.attn.c_proj.weight"

        w = tensors[w_name]
        b = tensors[b_name]
        p = tensors[p_name]

        if w.shape != (HIDDEN, 3 * HIDDEN):
            sys.exit(f"{w_name}: expected shape {(HIDDEN, 3 * HIDDEN)}, got {tuple(w.shape)}")
        if b.shape != (3 * HIDDEN,):
            sys.exit(f"{b_name}: expected shape {(3 * HIDDEN,)}, got {tuple(b.shape)}")
        if p.shape != (HIDDEN, HIDDEN):
            sys.exit(f"{p_name}: expected shape {(HIDDEN, HIDDEN)}, got {tuple(p.shape)}")

        # c_attn columns are [q(768) | k(768) | v(768)]; apply the same
        # per-head mask inside each 768-wide segment, in q, k, v order.
        col_mask = torch.cat([head_mask, head_mask, head_mask])
        out[w_name] = w[:, col_mask].contiguous()
        out[b_name] = b[col_mask].contiguous()

        # c_proj rows are the single 768-wide head-concatenated input.
        out[p_name] = p[head_mask, :].contiguous()

    # Required checks: fail loudly before writing.
    exp_attn_w = (HIDDEN, kept_dim * 3)
    exp_attn_b = (kept_dim * 3,)
    exp_proj_w = (kept_dim, HIDDEN)

    if out["h.0.attn.c_attn.weight"].shape != torch.Size(exp_attn_w):
        sys.exit("check failed: h.0.attn.c_attn.weight shape")
    if out["h.0.attn.c_attn.bias"].shape != torch.Size(exp_attn_b):
        sys.exit("check failed: h.0.attn.c_attn.bias shape")
    if out["h.0.attn.c_proj.weight"].shape != torch.Size(exp_proj_w):
        sys.exit("check failed: h.0.attn.c_proj.weight shape")
    if len(out) != 160:
        sys.exit(f"check failed: output has {len(out)} tensors, expected 160")

    assert exp_attn_w == (768, 2112)
    assert exp_attn_b == (2112,)
    assert exp_proj_w == (704, 768)

    # Spot-check against the exact column/row ranges given in the task spec
    # for layer 0, as an extra guard beyond the mask-based derivation above.
    spec_ranges = [(0, 320), (384, 768), (768, 1088), (1152, 1536), (1536, 1856), (1920, 2304)]
    expected_cols = torch.cat([torch.arange(a, b) for a, b in spec_ranges])
    actual_cols = torch.nonzero(torch.cat([head_mask, head_mask, head_mask])).squeeze(1)
    if not torch.equal(expected_cols, actual_cols):
        sys.exit("check failed: c_attn column selection does not match spec ranges")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT_PATH)
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
