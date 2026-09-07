"""T2: structured attention-head pruning of OLMo-1B-0724-hf.

Removes head 5 from every layer at the checkpoint level, using plain
torch + safetensors slicing.  Layout facts (from TASK.md and config.json):
16 layers, 16 heads, head_dim 128, hidden 2048, q/k/v/o unfused, all
matrices in nn.Linear [out, in] layout.  Heads are row blocks of q/k/v
and column blocks of o_proj.

All checks run before anything is written; the script raises on failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path("inputs/base")
OUT_DIR = Path("out/T2")
OUT_FILE = OUT_DIR / "model.safetensors"

N_LAYERS = 16
N_HEADS = 16
HEAD_DIM = 128
HIDDEN = N_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5
N_TENSORS = 114

ROW_TENSORS = ("q_proj", "k_proj", "v_proj")  # heads are row blocks
COL_TENSORS = ("o_proj",)                     # heads are column blocks


def check(cond: object, msg: str) -> None:
    """Fail loudly: no partial output is ever written on a failed check."""
    if not cond:
        raise AssertionError(msg)


def keep_slices() -> tuple[slice, slice]:
    lo = PRUNE_HEAD * HEAD_DIM          # 640
    hi = (PRUNE_HEAD + 1) * HEAD_DIM    # 768
    return slice(0, lo), slice(hi, HIDDEN)


def load_input() -> tuple[dict[str, torch.Tensor], list[str]]:
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    order = list(weight_map)  # deterministic: index order

    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(BASE / shard, framework="pt") as f:
            for key in f.keys():
                check(key in weight_map, f"{key} in {shard} but not in index")
                check(key not in tensors, f"duplicate tensor across shards: {key}")
                tensors[key] = f.get_tensor(key)

    missing = set(weight_map) - set(tensors)
    check(not missing, f"index lists tensors absent from the shards: {sorted(missing)}")
    check(len(tensors) == N_TENSORS, f"input has {len(tensors)} tensors, expected {N_TENSORS}")
    return tensors, order


def prune(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    head, tail = keep_slices()
    out: dict[str, torch.Tensor] = {}
    touched: set[str] = set()

    for i in range(N_LAYERS):
        for name in ROW_TENSORS:
            key = f"model.layers.{i}.self_attn.{name}.weight"
            t = src[key]
            check(tuple(t.shape) == (HIDDEN, HIDDEN), f"{key}: shape {tuple(t.shape)} != (2048, 2048)")
            out[key] = torch.cat([t[head, :], t[tail, :]], dim=0).contiguous()
            touched.add(key)
        for name in COL_TENSORS:
            key = f"model.layers.{i}.self_attn.{name}.weight"
            t = src[key]
            check(tuple(t.shape) == (HIDDEN, HIDDEN), f"{key}: shape {tuple(t.shape)} != (2048, 2048)")
            out[key] = torch.cat([t[:, head], t[:, tail]], dim=1).contiguous()
            touched.add(key)

    for key, t in src.items():
        if key not in touched:
            check("self_attn" not in key, f"unexpected head-bearing tensor left untouched: {key}")
            out[key] = t
    return out


def verify(src: dict[str, torch.Tensor], out: dict[str, torch.Tensor]) -> None:
    head, tail = keep_slices()
    kept_rows = torch.cat([torch.arange(head.start, head.stop), torch.arange(tail.start, tail.stop)])
    check(kept_rows.numel() == HIDDEN - HEAD_DIM, "kept index count is wrong")

    # Required checks, stated explicitly for layer 0.
    for name in ROW_TENSORS:
        key = f"model.layers.0.self_attn.{name}.weight"
        check(tuple(out[key].shape) == (1920, 2048), f"{key}: shape {tuple(out[key].shape)} != (1920, 2048)")
    key0 = "model.layers.0.self_attn.o_proj.weight"
    check(tuple(out[key0].shape) == (2048, 1920), f"{key0}: shape {tuple(out[key0].shape)} != (2048, 1920)")
    check(len(out) == N_TENSORS, f"output has {len(out)} tensors, expected {N_TENSORS}")

    # Same checks, plus bit-exactness and dtype, on every layer.
    check(set(out) == set(src), "output key set differs from input key set")
    for i in range(N_LAYERS):
        for name in ROW_TENSORS:
            key = f"model.layers.{i}.self_attn.{name}.weight"
            t = out[key]
            check(tuple(t.shape) == (1920, 2048), f"{key}: shape {tuple(t.shape)} != (1920, 2048)")
            check(torch.equal(t, src[key].index_select(0, kept_rows)), f"{key}: values are not the kept rows")
        key = f"model.layers.{i}.self_attn.o_proj.weight"
        t = out[key]
        check(tuple(t.shape) == (2048, 1920), f"{key}: shape {tuple(t.shape)} != (2048, 1920)")
        check(torch.equal(t, src[key].index_select(1, kept_rows)), f"{key}: values are not the kept columns")

    for key in out:
        check(out[key].dtype == src[key].dtype, f"{key}: dtype changed")
        check(out[key].is_contiguous(), f"{key}: not contiguous, safetensors would reject it")
        if "self_attn" not in key:
            check(torch.equal(out[key], src[key]), f"{key}: should have been left unchanged")
            check(out[key].shape == src[key].shape, f"{key}: shape changed but should not have")


def main() -> int:
    src, order = load_input()
    out = prune(src)
    verify(src, out)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file({k: out[k] for k in order}, str(OUT_FILE))

    # Post-write readback: the file on disk is what the checks passed on.
    with safe_open(OUT_FILE, framework="pt") as f:
        keys = list(f.keys())
        check(len(keys) == N_TENSORS, f"written file has {len(keys)} tensors, expected {N_TENSORS}")
        check(set(keys) == set(out), "written file key set differs")
        for key in keys:
            check(torch.equal(f.get_tensor(key), out[key]), f"{key}: readback mismatch")

    print(f"wrote {OUT_FILE} with {len(out)} tensors")
    for name, shape in (("q_proj", (1920, 2048)), ("k_proj", (1920, 2048)),
                        ("v_proj", (1920, 2048)), ("o_proj", (2048, 1920))):
        key = f"model.layers.0.self_attn.{name}.weight"
        print(f"  {key}: {tuple(out[key].shape)} (expected {shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
