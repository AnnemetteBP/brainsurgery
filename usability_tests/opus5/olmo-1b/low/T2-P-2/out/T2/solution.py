"""T2: structured attention-head pruning for OLMo-1B-0724-hf (remove head 5)."""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T2"
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
PRUNE_HEAD = 5

START = PRUNE_HEAD * HEAD_DIM          # 640
STOP = (PRUNE_HEAD + 1) * HEAD_DIM     # 768
KEEP = HIDDEN - HEAD_DIM               # 1920

ROW_PRUNED = {"q_proj", "k_proj", "v_proj"}
COL_PRUNED = {"o_proj"}


def load_all() -> dict[str, torch.Tensor]:
    with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    tensors: dict[str, torch.Tensor] = {}
    shards: dict[str, list[str]] = {}
    for name, shard in index["weight_map"].items():
        shards.setdefault(shard, []).append(name)
    for shard, names in shards.items():
        with safe_open(os.path.join(IN_DIR, shard), framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name)
    return tensors


def drop_block(t: torch.Tensor, dim: int) -> torch.Tensor:
    keep = torch.cat([t.narrow(dim, 0, START), t.narrow(dim, STOP, HIDDEN - STOP)], dim=dim)
    return keep.contiguous().clone()


def main() -> None:
    src = load_all()
    if len(src) != 114:
        raise SystemExit(f"expected 114 input tensors, got {len(src)}")

    out: dict[str, torch.Tensor] = {}
    touched = 0
    for name, t in src.items():
        parts = name.split(".")
        proj = parts[-2] if len(parts) >= 2 else ""
        is_attn = len(parts) >= 3 and parts[-3] == "self_attn"
        if is_attn and proj in ROW_PRUNED:
            if tuple(t.shape) != (HIDDEN, HIDDEN):
                raise SystemExit(f"{name}: unexpected shape {tuple(t.shape)}")
            out[name] = drop_block(t, 0)
            touched += 1
        elif is_attn and proj in COL_PRUNED:
            if tuple(t.shape) != (HIDDEN, HIDDEN):
                raise SystemExit(f"{name}: unexpected shape {tuple(t.shape)}")
            out[name] = drop_block(t, 1)
            touched += 1
        else:
            out[name] = t

    expected_touched = NUM_LAYERS * 4
    if touched != expected_touched:
        raise SystemExit(f"pruned {touched} tensors, expected {expected_touched}")

    # Required checks.
    for proj in ("q_proj", "k_proj", "v_proj"):
        key = f"model.layers.0.self_attn.{proj}.weight"
        if tuple(out[key].shape) != (KEEP, HIDDEN):
            raise SystemExit(f"{key}: shape {tuple(out[key].shape)} != ({KEEP}, {HIDDEN})")
    okey = "model.layers.0.self_attn.o_proj.weight"
    if tuple(out[okey].shape) != (HIDDEN, KEEP):
        raise SystemExit(f"{okey}: shape {tuple(out[okey].shape)} != ({HIDDEN}, {KEEP})")
    if len(out) != 114:
        raise SystemExit(f"output has {len(out)} tensors, expected 114")

    # Every layer must be consistent, not just layer 0.
    for i in range(NUM_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj"):
            key = f"model.layers.{i}.self_attn.{proj}.weight"
            if tuple(out[key].shape) != (KEEP, HIDDEN):
                raise SystemExit(f"{key}: shape {tuple(out[key].shape)}")
        key = f"model.layers.{i}.self_attn.o_proj.weight"
        if tuple(out[key].shape) != (HIDDEN, KEEP):
            raise SystemExit(f"{key}: shape {tuple(out[key].shape)}")

    # Dtypes and untouched tensors must be preserved exactly.
    for name, t in out.items():
        if t.dtype != src[name].dtype:
            raise SystemExit(f"{name}: dtype changed")

    # Spot-check the kept slices are bit-identical to the source rows/columns.
    ref = src["model.layers.0.self_attn.q_proj.weight"]
    got = out["model.layers.0.self_attn.q_proj.weight"]
    if not torch.equal(got[:START], ref[:START]) or not torch.equal(got[START:], ref[STOP:]):
        raise SystemExit("q_proj row order is wrong")
    ref = src["model.layers.0.self_attn.o_proj.weight"]
    got = out["model.layers.0.self_attn.o_proj.weight"]
    if not torch.equal(got[:, :START], ref[:, :START]) or not torch.equal(
        got[:, START:], ref[:, STOP:]
    ):
        raise SystemExit("o_proj column order is wrong")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, OUT_PATH, metadata={"format": "pt"})
    print(f"wrote {OUT_PATH}: {len(out)} tensors, pruned head {PRUNE_HEAD} in {NUM_LAYERS} layers")


if __name__ == "__main__":
    main()
