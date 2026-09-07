"""
T2: Structured attention-head pruning (Pythia-1B), condition F.

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). No
merge-config or prune_heads route was used: `transformers.prune_heads` targets
the *encoder* self-attention API and doesn't apply to GPTNeoX's fused
query_key_value projection, and mergekit's passthrough/task-arithmetic recipes
operate on whole layers/tensors, not on sub-tensor head slices. The layout is
fully specified in TASK.md (interleaved per-head 768-row QKV blocks, 256-wide
column blocks in `dense.weight`), so direct index-slicing is the smallest
correct tool for the job -- and it lets every required check run as a
plain assertion against measured tensor shapes before anything is written.

Removes head 5 (0-indexed) of 8 from every layer of Pythia-1B by slicing:
  - attention.query_key_value.weight: drop rows [3840:4608)   (768-row block)
  - attention.query_key_value.bias:   drop rows [3840:4608)
  - attention.dense.weight:           drop cols [1280:1536)   (256-col block)
All other tensors are copied through unchanged, byte-for-byte (same dtype,
no recompute).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HEAD_TO_PRUNE = 5
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN_SIZE = 2048
NUM_LAYERS = 16
QKV_BLOCK = 3 * HEAD_DIM  # 768: one head's q,k,v rows in the fused projection

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T2/model.safetensors")


def qkv_row_keep_slices(head: int) -> list[slice]:
    """Rows to keep in query_key_value.{weight,bias} when dropping `head`."""
    start = head * QKV_BLOCK
    end = start + QKV_BLOCK
    total = NUM_HEADS * QKV_BLOCK
    slices = []
    if start > 0:
        slices.append(slice(0, start))
    if end < total:
        slices.append(slice(end, total))
    return slices


def dense_col_keep_slices(head: int) -> list[slice]:
    """Columns to keep in dense.weight when dropping `head`."""
    start = head * HEAD_DIM
    end = start + HEAD_DIM
    total = NUM_HEADS * HEAD_DIM
    slices = []
    if start > 0:
        slices.append(slice(0, start))
    if end < total:
        slices.append(slice(end, total))
    return slices


def main() -> None:
    if not IN_PATH.is_file():
        sys.exit(f"input checkpoint not found: {IN_PATH}")

    state_dict = load_file(str(IN_PATH))
    print(f"loaded {len(state_dict)} tensors from {IN_PATH}")

    row_slices = qkv_row_keep_slices(HEAD_TO_PRUNE)
    col_slices = dense_col_keep_slices(HEAD_TO_PRUNE)
    assert row_slices == [slice(0, 3840), slice(4608, 6144)]
    assert col_slices == [slice(0, 1280), slice(1536, 2048)]

    out: dict[str, torch.Tensor] = {}
    touched = set()

    for i in range(NUM_LAYERS):
        qkv_w_key = f"gpt_neox.layers.{i}.attention.query_key_value.weight"
        qkv_b_key = f"gpt_neox.layers.{i}.attention.query_key_value.bias"
        dense_w_key = f"gpt_neox.layers.{i}.attention.dense.weight"

        qkv_w = state_dict[qkv_w_key]
        qkv_b = state_dict[qkv_b_key]
        dense_w = state_dict[dense_w_key]

        assert qkv_w.shape == (6144, HIDDEN_SIZE), (qkv_w_key, qkv_w.shape)
        assert qkv_b.shape == (6144,), (qkv_b_key, qkv_b.shape)
        assert dense_w.shape == (HIDDEN_SIZE, HIDDEN_SIZE), (dense_w_key, dense_w.shape)

        new_qkv_w = torch.cat([qkv_w[s] for s in row_slices], dim=0).contiguous()
        new_qkv_b = torch.cat([qkv_b[s] for s in row_slices], dim=0).contiguous()
        new_dense_w = torch.cat([dense_w[:, s] for s in col_slices], dim=1).contiguous()

        assert new_qkv_w.shape == (5376, HIDDEN_SIZE), (qkv_w_key, new_qkv_w.shape)
        assert new_qkv_b.shape == (5376,), (qkv_b_key, new_qkv_b.shape)
        assert new_dense_w.shape == (HIDDEN_SIZE, 1792), (dense_w_key, new_dense_w.shape)
        assert new_qkv_w.dtype == qkv_w.dtype
        assert new_qkv_b.dtype == qkv_b.dtype
        assert new_dense_w.dtype == dense_w.dtype

        out[qkv_w_key] = new_qkv_w
        out[qkv_b_key] = new_qkv_b
        out[dense_w_key] = new_dense_w
        touched.update({qkv_w_key, qkv_b_key, dense_w_key})

    for key, tensor in state_dict.items():
        if key not in touched:
            out[key] = tensor

    # Required checks (fail loudly before writing).
    assert out["gpt_neox.layers.0.attention.query_key_value.weight"].shape == (5376, 2048)
    assert out["gpt_neox.layers.0.attention.query_key_value.bias"].shape == (5376,)
    assert out["gpt_neox.layers.0.attention.dense.weight"].shape == (2048, 1792)
    assert len(out) == 244, f"expected 244 tensors, got {len(out)}"
    assert set(out.keys()) == set(state_dict.keys()), "tensor names must not change"

    # Untouched-tensor sanity: bit-identical to the input.
    for key in set(state_dict.keys()) - touched:
        assert torch.equal(out[key], state_dict[key]), f"unexpectedly modified: {key}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {len(out)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
