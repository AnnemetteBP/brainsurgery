"""
Prune head 5 (0-indexed) from every attention layer of Pythia-1B.

Pythia-1B (GPT-NeoX) attention layout:
- 8 heads, 256 dims/head, hidden size 2048.
- query_key_value.weight: [6144, 2048], rows grouped per head in 768-row
  blocks (head h -> rows 768*h .. 768*h+767); within a head's block the
  rows are [query(256) | key(256) | value(256)] (GPT-NeoX interleaved-by-head
  layout, not global [q|k|v] segments).
- query_key_value.bias: [6144], same per-head row layout.
- dense.weight: [2048, 2048], nn.Linear [out, in]; the *input* side (columns)
  is indexed by head in 256-wide blocks (dense consumes the concatenated
  per-head attention outputs).
- dense.bias: [2048], not per-head (it's on the output side), untouched.

To drop head 5 of 8, we keep every row/column block except the one belonging
to head 5.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HEAD_TO_PRUNE = 5
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN_SIZE = 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in the fused qkv projection
NUM_LAYERS = 16

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T2")
OUT_PATH = OUT_DIR / "model.safetensors"


def kept_row_ranges_qkv():
    """Rows to keep in query_key_value.{weight,bias}: every head's 768-row
    block except the pruned head's, in original order."""
    starts = [h * QKV_BLOCK for h in range(NUM_HEADS) if h != HEAD_TO_PRUNE]
    idx = torch.cat([torch.arange(s, s + QKV_BLOCK) for s in starts])
    return idx


def kept_col_ranges_dense():
    """Columns to keep in dense.weight: every head's 256-col block except the
    pruned head's, in original order."""
    starts = [h * HEAD_DIM for h in range(NUM_HEADS) if h != HEAD_TO_PRUNE]
    idx = torch.cat([torch.arange(s, s + HEAD_DIM) for s in starts])
    return idx


def main():
    if not IN_PATH.exists():
        sys.exit(f"missing input: {IN_PATH}")

    tensors = load_file(str(IN_PATH))

    expected_qkv_w = (NUM_HEADS * QKV_BLOCK, HIDDEN_SIZE)
    expected_dense_w = (HIDDEN_SIZE, HIDDEN_SIZE)
    qkv_w_key0 = "gpt_neox.layers.0.attention.query_key_value.weight"
    if tuple(tensors[qkv_w_key0].shape) != expected_qkv_w:
        sys.exit(
            f"unexpected input shape for {qkv_w_key0}: "
            f"{tuple(tensors[qkv_w_key0].shape)} != {expected_qkv_w}"
        )

    qkv_row_idx = kept_row_ranges_qkv()
    dense_col_idx = kept_col_ranges_dense()

    out = dict(tensors)  # start from a shallow copy; overwrite per-head tensors

    for i in range(NUM_LAYERS):
        qkv_w_key = f"gpt_neox.layers.{i}.attention.query_key_value.weight"
        qkv_b_key = f"gpt_neox.layers.{i}.attention.query_key_value.bias"
        dense_w_key = f"gpt_neox.layers.{i}.attention.dense.weight"

        for key in (qkv_w_key, qkv_b_key, dense_w_key):
            if key not in tensors:
                sys.exit(f"missing expected tensor: {key}")

        qkv_w = tensors[qkv_w_key]
        qkv_b = tensors[qkv_b_key]
        dense_w = tensors[dense_w_key]

        if tuple(qkv_w.shape) != expected_qkv_w:
            sys.exit(f"{qkv_w_key}: shape {tuple(qkv_w.shape)} != {expected_qkv_w}")
        if tuple(qkv_b.shape) != (expected_qkv_w[0],):
            sys.exit(f"{qkv_b_key}: shape {tuple(qkv_b.shape)} != ({expected_qkv_w[0]},)")
        if tuple(dense_w.shape) != expected_dense_w:
            sys.exit(f"{dense_w_key}: shape {tuple(dense_w.shape)} != {expected_dense_w}")

        new_qkv_w = qkv_w.index_select(0, qkv_row_idx).contiguous()
        new_qkv_b = qkv_b.index_select(0, qkv_row_idx).contiguous()
        new_dense_w = dense_w.index_select(1, dense_col_idx).contiguous()

        out[qkv_w_key] = new_qkv_w
        out[qkv_b_key] = new_qkv_b
        out[dense_w_key] = new_dense_w

    # --- required checks ---
    exp_qkv_w = (5376, 2048)
    exp_qkv_b = (5376,)
    exp_dense_w = (2048, 1792)

    got = tuple(out["gpt_neox.layers.0.attention.query_key_value.weight"].shape)
    if got != exp_qkv_w:
        sys.exit(f"check failed: layer 0 qkv.weight shape {got} != {exp_qkv_w}")

    got = tuple(out["gpt_neox.layers.0.attention.query_key_value.bias"].shape)
    if got != exp_qkv_b:
        sys.exit(f"check failed: layer 0 qkv.bias shape {got} != {exp_qkv_b}")

    got = tuple(out["gpt_neox.layers.0.attention.dense.weight"].shape)
    if got != exp_dense_w:
        sys.exit(f"check failed: layer 0 dense.weight shape {got} != {exp_dense_w}")

    if len(out) != 244:
        sys.exit(f"check failed: output has {len(out)} tensors, expected 244")

    # sanity: verify the row-selection is exactly rows 0..3839 + 4608..6143,
    # and columns 0..1279 + 1536..2047, as spelled out in the task.
    expected_qkv_idx = torch.cat([torch.arange(0, 3840), torch.arange(4608, 6144)])
    if not torch.equal(qkv_row_idx, expected_qkv_idx):
        sys.exit("check failed: qkv row selection does not match spec")
    expected_dense_idx = torch.cat([torch.arange(0, 1280), torch.arange(1536, 2048)])
    if not torch.equal(dense_col_idx, expected_dense_idx):
        sys.exit("check failed: dense column selection does not match spec")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
