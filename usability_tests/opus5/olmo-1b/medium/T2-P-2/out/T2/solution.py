"""T2: structured attention-head pruning of OLMo-1B-0724-hf.

Removes head 5 (of 16 heads x 128 dims) from every layer's q/k/v (row blocks)
and o_proj (column blocks), writing a single-file safetensors checkpoint.
"""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "..", "inputs", "base")
OUT_DIR = os.path.join(HERE)
OUT_FILE = os.path.join(OUT_DIR, "model.safetensors")

NUM_HEADS = 16
HEAD_DIM = 128
PRUNE_HEAD = 5
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
LO = PRUNE_HEAD * HEAD_DIM      # 640
HI = LO + HEAD_DIM              # 768
KEPT = HIDDEN - HEAD_DIM        # 1920

ROW_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.(q|k|v)_proj\.weight$")
COL_RE = re.compile(r"^model\.layers\.\d+\.self_attn\.o_proj\.weight$")

keep = torch.cat([torch.arange(0, LO), torch.arange(HI, HIDDEN)])


def load_state_dict():
    index_path = os.path.join(IN_DIR, "model.safetensors.index.json")
    with open(index_path) as fh:
        weight_map = json.load(fh)["weight_map"]
    by_shard = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)
    sd = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(os.path.join(IN_DIR, shard), framework="pt", device="cpu") as fh:
            for name in names:
                sd[name] = fh.get_tensor(name)
    return sd


def main():
    sd = load_state_dict()
    n_in = len(sd)
    print(f"loaded {n_in} tensors")

    out = {}
    n_row = n_col = 0
    for name, t in sd.items():
        if ROW_RE.match(name):
            assert t.shape == (HIDDEN, HIDDEN), f"{name}: unexpected shape {tuple(t.shape)}"
            new = t.index_select(0, keep)
            assert new.shape == (KEPT, HIDDEN), f"{name}: got {tuple(new.shape)}"
            # value spot-check: the kept blocks must be verbatim slices of the input
            assert torch.equal(new[:LO], t[:LO])
            assert torch.equal(new[LO:], t[HI:])
            n_row += 1
        elif COL_RE.match(name):
            assert t.shape == (HIDDEN, HIDDEN), f"{name}: unexpected shape {tuple(t.shape)}"
            new = t.index_select(1, keep)
            assert new.shape == (HIDDEN, KEPT), f"{name}: got {tuple(new.shape)}"
            assert torch.equal(new[:, :LO], t[:, :LO])
            assert torch.equal(new[:, LO:], t[:, HI:])
            n_col += 1
        else:
            new = t
        assert new.dtype == t.dtype, f"{name}: dtype drift {t.dtype} -> {new.dtype}"
        # clone so nothing shares storage with the mmap or with another tensor
        out[name] = new.contiguous().clone()

    print(f"pruned {n_row} row-block tensors, {n_col} column-block tensors")
    assert n_row == 3 * 16, f"expected 48 q/k/v tensors, got {n_row}"
    assert n_col == 16, f"expected 16 o_proj tensors, got {n_col}"

    # Required checks, verbatim from TASK.md.
    assert tuple(out["model.layers.0.self_attn.q_proj.weight"].shape) == (1920, 2048)
    assert tuple(out["model.layers.0.self_attn.k_proj.weight"].shape) == (1920, 2048)
    assert tuple(out["model.layers.0.self_attn.v_proj.weight"].shape) == (1920, 2048)
    assert tuple(out["model.layers.0.self_attn.o_proj.weight"].shape) == (2048, 1920)
    assert len(out) == 114, f"expected 114 tensors, got {len(out)}"
    assert len(out) == n_in, f"tensor count changed: {n_in} -> {len(out)}"

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(out, OUT_FILE)
    print(f"wrote {OUT_FILE}")

    # Read back and re-verify what we just wrote.
    with safe_open(OUT_FILE, framework="pt", device="cpu") as fh:
        keys = list(fh.keys())
        assert len(keys) == 114, f"readback: {len(keys)} tensors"
        for k in ("q", "k", "v"):
            n = f"model.layers.0.self_attn.{k}_proj.weight"
            assert tuple(fh.get_slice(n).get_shape()) == (1920, 2048), n
        assert tuple(
            fh.get_slice("model.layers.0.self_attn.o_proj.weight").get_shape()
        ) == (2048, 1920)
    print("readback OK: 114 tensors, shapes as required")


if __name__ == "__main__":
    main()
