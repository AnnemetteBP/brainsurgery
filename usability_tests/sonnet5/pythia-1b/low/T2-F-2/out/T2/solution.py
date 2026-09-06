"""
T2: Structured attention-head pruning (Pythia-1B), remove head 5 of 8 in every layer.

Approach: direct tensor slicing with safetensors + torch. Chosen over
transformers' `prune_heads` because that API prunes heads by index within its
own internal q/k/v-split view and is not guaranteed to reproduce the exact
byte layout / slice boundaries specified in TASK.md (rows 0..3839,
4608..6143 for qkv; columns 0..1279, 1536..2047 for dense). A plain script
slicing the raw state dict rows/columns is unambiguous and directly
verifiable against the spec, and lets us assert every required shape before
writing.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T2/model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in fused qkv
PRUNE_HEAD = 5

EXPECTED_QKV_W_SHAPE = (5376, 2048)
EXPECTED_QKV_B_SHAPE = (5376,)
EXPECTED_DENSE_W_SHAPE = (2048, 1792)
EXPECTED_NUM_TENSORS = 244


def keep_row_ranges(total_heads, prune_head, block):
    """Row index ranges to keep when removing one head's block of `block` rows."""
    lo = prune_head * block
    hi = lo + block
    return lo, hi


def main():
    state_dict = load_file(str(IN_PATH))
    if len(state_dict) != EXPECTED_NUM_TENSORS:
        sys.exit(
            f"input tensor count mismatch: got {len(state_dict)}, "
            f"expected {EXPECTED_NUM_TENSORS}"
        )

    out = {}
    for i in range(NUM_LAYERS):
        qkv_w_key = f"gpt_neox.layers.{i}.attention.query_key_value.weight"
        qkv_b_key = f"gpt_neox.layers.{i}.attention.query_key_value.bias"
        dense_w_key = f"gpt_neox.layers.{i}.attention.dense.weight"

        qkv_w = state_dict.pop(qkv_w_key)
        qkv_b = state_dict.pop(qkv_b_key)
        dense_w = state_dict.pop(dense_w_key)

        lo, hi = keep_row_ranges(NUM_HEADS, PRUNE_HEAD, QKV_BLOCK)
        assert (lo, hi) == (3840, 4608), (lo, hi)

        new_qkv_w = torch.cat([qkv_w[:lo], qkv_w[hi:]], dim=0)
        new_qkv_b = torch.cat([qkv_b[:lo], qkv_b[hi:]], dim=0)

        col_lo, col_hi = keep_row_ranges(NUM_HEADS, PRUNE_HEAD, HEAD_DIM)
        assert (col_lo, col_hi) == (1280, 1536), (col_lo, col_hi)
        new_dense_w = torch.cat([dense_w[:, :col_lo], dense_w[:, col_hi:]], dim=1)

        if new_qkv_w.shape != EXPECTED_QKV_W_SHAPE:
            sys.exit(f"layer {i}: qkv weight shape {new_qkv_w.shape}, expected {EXPECTED_QKV_W_SHAPE}")
        if new_qkv_b.shape != EXPECTED_QKV_B_SHAPE:
            sys.exit(f"layer {i}: qkv bias shape {new_qkv_b.shape}, expected {EXPECTED_QKV_B_SHAPE}")
        if new_dense_w.shape != EXPECTED_DENSE_W_SHAPE:
            sys.exit(f"layer {i}: dense weight shape {new_dense_w.shape}, expected {EXPECTED_DENSE_W_SHAPE}")

        out[qkv_w_key] = new_qkv_w.contiguous()
        out[qkv_b_key] = new_qkv_b.contiguous()
        out[dense_w_key] = new_dense_w.contiguous()

    # Everything else (dense.bias, MLP tensors, attention buffers, embeddings, etc.)
    # passes through unchanged.
    out.update(state_dict)

    if len(out) != EXPECTED_NUM_TENSORS:
        sys.exit(f"output tensor count {len(out)}, expected {EXPECTED_NUM_TENSORS}")

    l0_qkv_w = out["gpt_neox.layers.0.attention.query_key_value.weight"]
    l0_qkv_b = out["gpt_neox.layers.0.attention.query_key_value.bias"]
    l0_dense_w = out["gpt_neox.layers.0.attention.dense.weight"]
    assert tuple(l0_qkv_w.shape) == EXPECTED_QKV_W_SHAPE
    assert tuple(l0_qkv_b.shape) == EXPECTED_QKV_B_SHAPE
    assert tuple(l0_dense_w.shape) == EXPECTED_DENSE_W_SHAPE

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
