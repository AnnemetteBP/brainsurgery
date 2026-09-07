"""Prune head 5 from every attention layer of Pythia-1B.

Fused query_key_value: 6144 rows = 8 heads * 768 rows/head (q256,k256,v256).
Removing head 5 keeps rows 0..3839 and 4608..6143 (768*5=3840, 768*6=4608).

dense.weight: [2048 out, 2048 in], heads are 256-wide column (input) blocks.
Removing head 5 keeps columns 0..1279 and 1536..2047 (256*5=1280, 256*6=1536).
"""

import torch
from safetensors.torch import load_file, save_file

NUM_LAYERS = 16
HEAD_DIM = 256
QKV_HEAD_BLOCK = 3 * HEAD_DIM  # 768
HEAD_TO_REMOVE = 5

IN_PATH = "inputs/base/model.safetensors"
OUT_PATH = "out/T2/model.safetensors"


def drop_rows(t: torch.Tensor, block: int, head: int) -> torch.Tensor:
    lo = block * head
    hi = block * (head + 1)
    return torch.cat([t[:lo], t[hi:]], dim=0)


def drop_cols(t: torch.Tensor, block: int, head: int) -> torch.Tensor:
    lo = block * head
    hi = block * (head + 1)
    return torch.cat([t[:, :lo], t[:, hi:]], dim=1)


def main() -> None:
    state = load_file(IN_PATH)
    assert len(state) == 244, f"expected 244 input tensors, got {len(state)}"

    out = dict(state)

    for i in range(NUM_LAYERS):
        wk = f"gpt_neox.layers.{i}.attention.query_key_value.weight"
        bk = f"gpt_neox.layers.{i}.attention.query_key_value.bias"
        dk = f"gpt_neox.layers.{i}.attention.dense.weight"

        qkv_w = state[wk]
        qkv_b = state[bk]
        dense_w = state[dk]

        assert qkv_w.shape == (6144, 2048), f"{wk} unexpected shape {qkv_w.shape}"
        assert qkv_b.shape == (6144,), f"{bk} unexpected shape {qkv_b.shape}"
        assert dense_w.shape == (2048, 2048), f"{dk} unexpected shape {dense_w.shape}"

        out[wk] = drop_rows(qkv_w, QKV_HEAD_BLOCK, HEAD_TO_REMOVE)
        out[bk] = drop_rows(qkv_b, QKV_HEAD_BLOCK, HEAD_TO_REMOVE)
        out[dk] = drop_cols(dense_w, HEAD_DIM, HEAD_TO_REMOVE)

    # Required checks
    w0 = out["gpt_neox.layers.0.attention.query_key_value.weight"]
    b0 = out["gpt_neox.layers.0.attention.query_key_value.bias"]
    d0 = out["gpt_neox.layers.0.attention.dense.weight"]

    assert tuple(w0.shape) == (5376, 2048), f"bad qkv weight shape: {w0.shape}"
    assert tuple(b0.shape) == (5376,), f"bad qkv bias shape: {b0.shape}"
    assert tuple(d0.shape) == (2048, 1792), f"bad dense weight shape: {d0.shape}"
    assert len(out) == 244, f"expected 244 output tensors, got {len(out)}"

    # sanity: only per-head tensors changed shape from the input
    for k, v in out.items():
        if k not in state:
            raise AssertionError(f"unexpected new key {k}")
        if v.shape != state[k].shape:
            expected_changed = k.split(".")[-2:] in (
                ["query_key_value", "weight"],
                ["query_key_value", "bias"],
            ) or k.endswith("attention.dense.weight")
            if not expected_changed:
                raise AssertionError(f"unexpected shape change for {k}")
        out[k] = v.contiguous()

    save_file(out, OUT_PATH)
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
