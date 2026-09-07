"""T2: structured attention-head pruning for Pythia-1B (GPT-NeoX).

Removes head 5 from every layer at the checkpoint level.

Layout facts this relies on (from TASK.md, verified against the input):
  * `attention.query_key_value.weight` is `[3*hidden, hidden]` in nn.Linear
    `[out, in]` order, with rows grouped *per head*: head h owns the 768-row
    block `[768*h, 768*(h+1))`, holding its q, then k, then v (GPT-NeoX
    interleaved layout -- NOT `[q | k | v]` segments).
  * `attention.query_key_value.bias` is `[3*hidden]` with the same row layout.
  * `attention.dense.weight` is `[hidden, hidden]`; the input axis (columns)
    is head-major, so head h owns columns `[256*h, 256*(h+1))`.

Everything else (dense.bias, the attention buffers, the MLP and embeddings)
is copied through byte-for-byte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T2")
OUT_PATH = OUT_DIR / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5

QKV_BLOCK = 3 * HEAD_DIM  # 768 rows of fused qkv per head


def keep_index(num_blocks: int, block: int, drop: int) -> torch.Tensor:
    """Indices of every block except `drop`, in ascending order."""
    keep = [h for h in range(num_blocks) if h != drop]
    return torch.cat([torch.arange(h * block, (h + 1) * block) for h in keep])


def main() -> int:
    if not IN_PATH.exists():
        raise SystemExit(f"missing input: {IN_PATH}")

    sd = load_file(str(IN_PATH))
    n_in = len(sd)

    qkv_keep = keep_index(NUM_HEADS, QKV_BLOCK, PRUNE_HEAD)
    dense_keep = keep_index(NUM_HEADS, HEAD_DIM, PRUNE_HEAD)

    # Guard the index sets against the literal ranges spelled out in TASK.md,
    # so a wrong block boundary fails here and not silently in the output.
    expected_qkv = torch.cat([torch.arange(0, 3840), torch.arange(4608, 6144)])
    expected_dense = torch.cat([torch.arange(0, 1280), torch.arange(1536, 2048)])
    assert torch.equal(qkv_keep, expected_qkv), "qkv row selection != rows 0..3839, 4608..6143"
    assert torch.equal(dense_keep, expected_dense), "dense col selection != cols 0..1279, 1536..2047"

    touched = 0
    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention"
        w, b, d = f"{p}.query_key_value.weight", f"{p}.query_key_value.bias", f"{p}.dense.weight"
        for k in (w, b, d):
            if k not in sd:
                raise SystemExit(f"expected tensor missing from input: {k}")
        if tuple(sd[w].shape) != (3 * HIDDEN, HIDDEN):
            raise SystemExit(f"{w}: unexpected input shape {tuple(sd[w].shape)}")
        if tuple(sd[b].shape) != (3 * HIDDEN,):
            raise SystemExit(f"{b}: unexpected input shape {tuple(sd[b].shape)}")
        if tuple(sd[d].shape) != (HIDDEN, HIDDEN):
            raise SystemExit(f"{d}: unexpected input shape {tuple(sd[d].shape)}")

        sd[w] = sd[w].index_select(0, qkv_keep).contiguous()
        sd[b] = sd[b].index_select(0, qkv_keep).contiguous()
        sd[d] = sd[d].index_select(1, dense_keep).contiguous()
        touched += 3

    # --- Required checks: fail loudly before writing ------------------------
    kept_heads = NUM_HEADS - 1
    checks = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (kept_heads * QKV_BLOCK, HIDDEN),
        "gpt_neox.layers.0.attention.query_key_value.bias": (kept_heads * QKV_BLOCK,),
        "gpt_neox.layers.0.attention.dense.weight": (HIDDEN, kept_heads * HEAD_DIM),
    }
    for key, want in checks.items():
        got = tuple(sd[key].shape)
        if got != want:
            raise SystemExit(f"CHECK FAILED: {key} has shape {list(got)}, expected {list(want)}")
    if len(sd) != 244:
        raise SystemExit(f"CHECK FAILED: output has {len(sd)} tensors, expected 244")
    if len(sd) != n_in:
        raise SystemExit(f"CHECK FAILED: tensor count changed {n_in} -> {len(sd)}")

    # Every layer, not just layer 0.
    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention"
        for key, want in (
            (f"{p}.query_key_value.weight", (kept_heads * QKV_BLOCK, HIDDEN)),
            (f"{p}.query_key_value.bias", (kept_heads * QKV_BLOCK,)),
            (f"{p}.dense.weight", (HIDDEN, kept_heads * HEAD_DIM)),
        ):
            got = tuple(sd[key].shape)
            if got != want:
                raise SystemExit(f"CHECK FAILED: {key} has shape {list(got)}, expected {list(want)}")
        if tuple(sd[f"{p}.dense.bias"].shape) != (HIDDEN,):
            raise SystemExit(f"CHECK FAILED: {p}.dense.bias was modified")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(OUT_PATH), metadata={"format": "pt"})
    print(f"wrote {OUT_PATH}: {len(sd)} tensors, {touched} pruned ({NUM_LAYERS} layers x 3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
