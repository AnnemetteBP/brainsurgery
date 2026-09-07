"""
T2: Structured attention-head pruning (Pythia-1B).

Removes head 5 (0-indexed) from every layer's attention block:

  - attention.query_key_value.weight  [6144, 2048] -> [5376, 2048]
  - attention.query_key_value.bias    [6144]        -> [5376]
  - attention.dense.weight            [2048, 2048]  -> [2048, 1792]

Everything else is copied through unchanged. Layout notes (from TASK.md):
  - query_key_value rows are grouped per head in 768-row blocks (interleaved
    q/k/v within each block, GPT-NeoX style), so dropping a head means
    dropping its whole 768-row block: rows [768*h : 768*h+768).
  - dense.weight consumes the head outputs on its input (column) axis, in
    256-wide column blocks per head: columns [256*h : 256*h+256).
"""

import os

import torch
from safetensors.torch import load_file, save_file

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768: one head's q, k, v rows in the fused projection
HEAD_TO_PRUNE = 5

INPUT_PATH = "inputs/base/model.safetensors"
OUTPUT_DIR = "out/T2"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.safetensors")


def keep_row_mask(total_rows: int, block: int, drop_index: int) -> torch.Tensor:
    """Boolean mask over `total_rows`, dropping the [drop_index*block, (drop_index+1)*block) slice."""
    mask = torch.ones(total_rows, dtype=torch.bool)
    mask[drop_index * block : (drop_index + 1) * block] = False
    return mask


def prune_layer(state_dict: dict, layer: int) -> None:
    prefix = f"gpt_neox.layers.{layer}.attention"

    qkv_w_key = f"{prefix}.query_key_value.weight"
    qkv_b_key = f"{prefix}.query_key_value.bias"
    dense_w_key = f"{prefix}.dense.weight"

    qkv_w = state_dict[qkv_w_key]
    qkv_b = state_dict[qkv_b_key]
    dense_w = state_dict[dense_w_key]

    assert qkv_w.shape == (NUM_HEADS * QKV_BLOCK, HIDDEN), (
        f"{qkv_w_key}: unexpected input shape {tuple(qkv_w.shape)}"
    )
    assert qkv_b.shape == (NUM_HEADS * QKV_BLOCK,), (
        f"{qkv_b_key}: unexpected input shape {tuple(qkv_b.shape)}"
    )
    assert dense_w.shape == (HIDDEN, HIDDEN), (
        f"{dense_w_key}: unexpected input shape {tuple(dense_w.shape)}"
    )

    row_mask = keep_row_mask(NUM_HEADS * QKV_BLOCK, QKV_BLOCK, HEAD_TO_PRUNE)
    col_mask = keep_row_mask(NUM_HEADS * HEAD_DIM, HEAD_DIM, HEAD_TO_PRUNE)

    state_dict[qkv_w_key] = qkv_w[row_mask, :].contiguous()
    state_dict[qkv_b_key] = qkv_b[row_mask].contiguous()
    state_dict[dense_w_key] = dense_w[:, col_mask].contiguous()


def main() -> None:
    state_dict = load_file(INPUT_PATH)
    original_count = len(state_dict)

    for layer in range(NUM_LAYERS):
        prune_layer(state_dict, layer)

    # Required checks: fail loudly before writing anything.
    l0_qkv_w = state_dict["gpt_neox.layers.0.attention.query_key_value.weight"]
    l0_qkv_b = state_dict["gpt_neox.layers.0.attention.query_key_value.bias"]
    l0_dense_w = state_dict["gpt_neox.layers.0.attention.dense.weight"]

    assert l0_qkv_w.shape == (5376, 2048), (
        f"layer 0 query_key_value.weight: expected [5376, 2048], got {tuple(l0_qkv_w.shape)}"
    )
    assert l0_qkv_b.shape == (5376,), (
        f"layer 0 query_key_value.bias: expected [5376], got {tuple(l0_qkv_b.shape)}"
    )
    assert l0_dense_w.shape == (2048, 1792), (
        f"layer 0 dense.weight: expected [2048, 1792], got {tuple(l0_dense_w.shape)}"
    )
    assert len(state_dict) == original_count == 244, (
        f"expected 244 tensors, have {len(state_dict)} (input had {original_count})"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file(state_dict, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} with {len(state_dict)} tensors.")


if __name__ == "__main__":
    main()
