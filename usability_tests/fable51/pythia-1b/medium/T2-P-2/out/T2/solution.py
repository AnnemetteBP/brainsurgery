"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout)."""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"

N_LAYERS = 16
N_HEADS = 8
HEAD_DIM = 256
HIDDEN = N_HEADS * HEAD_DIM  # 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused projection
PRUNE_HEAD = 5


def drop_block(t: torch.Tensor, dim: int, block: int, idx: int) -> torch.Tensor:
    """Remove block `idx` of width `block` along `dim`, preserving order of the rest."""
    lo, hi = block * idx, block * (idx + 1)
    assert t.shape[dim] == block * N_HEADS, (t.shape, dim)
    return torch.cat([t.narrow(dim, 0, lo), t.narrow(dim, hi, t.shape[dim] - hi)], dim=dim).contiguous()


def main() -> None:
    sd = load_file(SRC)
    assert len(sd) == 244, f"expected 244 input tensors, got {len(sd)}"

    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
        assert sd[w].shape == (3 * HIDDEN, HIDDEN), sd[w].shape
        assert sd[b].shape == (3 * HIDDEN,), sd[b].shape
        assert sd[d].shape == (HIDDEN, HIDDEN), sd[d].shape
        sd[w] = drop_block(sd[w], 0, QKV_BLOCK, PRUNE_HEAD)
        sd[b] = drop_block(sd[b], 0, QKV_BLOCK, PRUNE_HEAD)
        sd[d] = drop_block(sd[d], 1, HEAD_DIM, PRUNE_HEAD)

    # Required checks (fail loudly before writing).
    p0 = "gpt_neox.layers.0.attention."
    checks = {
        p0 + "query_key_value.weight": (5376, 2048),
        p0 + "query_key_value.bias": (5376,),
        p0 + "dense.weight": (2048, 1792),
    }
    for k, shape in checks.items():
        if tuple(sd[k].shape) != shape:
            sys.exit(f"CHECK FAILED: {k} has shape {tuple(sd[k].shape)}, expected {shape}")
    if len(sd) != 244:
        sys.exit(f"CHECK FAILED: output has {len(sd)} tensors, expected 244")
    # Same checks on every layer, plus dtype preservation.
    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        assert sd[p + "query_key_value.weight"].shape == (5376, 2048)
        assert sd[p + "query_key_value.bias"].shape == (5376,)
        assert sd[p + "dense.weight"].shape == (2048, 1792)
        assert sd[p + "dense.weight"].dtype == torch.float16

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file(sd, DST)

    # Post-write verification.
    out = load_file(DST)
    assert len(out) == 244, len(out)
    for k, shape in checks.items():
        assert tuple(out[k].shape) == shape, (k, out[k].shape)
    print(f"OK: wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    main()
