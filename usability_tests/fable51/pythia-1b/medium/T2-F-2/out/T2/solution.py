"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout).

Plain safetensors + torch script: slice the per-head blocks out of the fused
query_key_value weight/bias (768-row blocks per head) and the dense output
projection (256-wide column blocks), leave every other tensor untouched.
"""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused projection


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def drop_block(t, dim, block, idx):
    """Remove block `idx` of size `block` along `dim`, keeping order."""
    n = t.shape[dim]
    check(n == block * NUM_HEADS, f"unexpected size {n} on dim {dim}")
    lo, hi = block * idx, block * (idx + 1)
    keep = torch.cat([t.narrow(dim, 0, lo), t.narrow(dim, hi, n - hi)], dim=dim)
    check(keep.shape[dim] == n - block, "slice size mismatch")
    return keep.contiguous()


def main():
    sd = load_file(SRC)
    check(len(sd) == 244, f"expected 244 input tensors, got {len(sd)}")
    out = dict(sd)

    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
        check(sd[w].shape == (3 * HIDDEN, HIDDEN), f"{w} shape {tuple(sd[w].shape)}")
        check(sd[b].shape == (3 * HIDDEN,), f"{b} shape {tuple(sd[b].shape)}")
        check(sd[d].shape == (HIDDEN, HIDDEN), f"{d} shape {tuple(sd[d].shape)}")
        out[w] = drop_block(sd[w], 0, QKV_BLOCK, PRUNE_HEAD)
        out[b] = drop_block(sd[b], 0, QKV_BLOCK, PRUNE_HEAD)
        out[d] = drop_block(sd[d], 1, HEAD_DIM, PRUNE_HEAD)

    # Required checks (fail loudly before writing).
    l0 = "gpt_neox.layers.0.attention."
    check(tuple(out[l0 + "query_key_value.weight"].shape) == (5376, 2048), "layer0 qkv weight shape")
    check(tuple(out[l0 + "query_key_value.bias"].shape) == (5376,), "layer0 qkv bias shape")
    check(tuple(out[l0 + "dense.weight"].shape) == (2048, 1792), "layer0 dense weight shape")
    check(len(out) == 244, f"expected 244 output tensors, got {len(out)}")
    # Extra sanity: keys, dtypes, and untouched tensors identical.
    check(set(out) == set(sd), "key set changed")
    for k in sd:
        check(out[k].dtype == sd[k].dtype, f"dtype changed for {k}")
        if not k.startswith("gpt_neox.layers.") or not (
            k.endswith("attention.query_key_value.weight")
            or k.endswith("attention.query_key_value.bias")
            or k.endswith("attention.dense.weight")
        ):
            check(out[k] is sd[k], f"unexpected modification of {k}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file(out, DST, metadata={"format": "pt"})
    print(f"wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    sys.exit(main())
