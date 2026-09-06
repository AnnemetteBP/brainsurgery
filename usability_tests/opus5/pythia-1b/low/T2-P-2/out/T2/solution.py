"""T2: structured head pruning for Pythia-1B (remove head 5 of 8 in every layer)."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
PRUNE = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 interleaved q|k|v rows per head


def keep_index(block: int, n_blocks: int, prune: int) -> torch.Tensor:
    keep = [i for i in range(n_blocks) if i != prune]
    return torch.cat([torch.arange(b * block, (b + 1) * block) for b in keep])


def main() -> None:
    sd = load_file(str(SRC))
    n_in = len(sd)

    qkv_idx = keep_index(QKV_BLOCK, NUM_HEADS, PRUNE)
    dense_idx = keep_index(HEAD_DIM, NUM_HEADS, PRUNE)

    out = dict(sd)
    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
        for k in (w, b, d):
            if k not in sd:
                raise KeyError(f"missing expected tensor: {k}")
        if sd[w].shape != (NUM_HEADS * QKV_BLOCK, 2048):
            raise ValueError(f"{w}: unexpected shape {tuple(sd[w].shape)}")
        if sd[b].shape != (NUM_HEADS * QKV_BLOCK,):
            raise ValueError(f"{b}: unexpected shape {tuple(sd[b].shape)}")
        if sd[d].shape != (2048, NUM_HEADS * HEAD_DIM):
            raise ValueError(f"{d}: unexpected shape {tuple(sd[d].shape)}")
        out[w] = sd[w].index_select(0, qkv_idx).contiguous()
        out[b] = sd[b].index_select(0, qkv_idx).contiguous()
        out[d] = sd[d].index_select(1, dense_idx).contiguous()

    # Required checks.
    checks = {
        "gpt_neox.layers.0.attention.query_key_value.weight": (5376, 2048),
        "gpt_neox.layers.0.attention.query_key_value.bias": (5376,),
        "gpt_neox.layers.0.attention.dense.weight": (2048, 1792),
    }
    for k, want in checks.items():
        got = tuple(out[k].shape)
        if got != want:
            raise AssertionError(f"{k}: shape {got}, expected {want}")
    if len(out) != 244:
        raise AssertionError(f"output has {len(out)} tensors, expected 244")
    if len(out) != n_in:
        raise AssertionError(f"tensor count changed: {n_in} -> {len(out)}")
    for k, v in out.items():
        if v.dtype != sd[k].dtype:
            raise AssertionError(f"{k}: dtype changed")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST))
    print(f"wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    main()
