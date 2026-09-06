"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout)."""

import os
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head in fused qkv
PRUNE_HEAD = 5


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def drop_block(t: torch.Tensor, dim: int, start: int, size: int) -> torch.Tensor:
    """Remove t[start:start+size] along `dim`, keeping the rest in order."""
    check(t.shape[dim] > start + size - 1, f"block {start}:{start+size} out of range for {tuple(t.shape)}")
    keep = torch.cat([t.narrow(dim, 0, start), t.narrow(dim, start + size, t.shape[dim] - start - size)], dim=dim)
    return keep.contiguous()


def main() -> None:
    check(os.path.isfile(SRC), f"missing input {SRC}")
    sd = load_file(SRC)
    check(len(sd) == 244, f"expected 244 input tensors, got {len(sd)}")

    out = dict(sd)
    for i in range(NUM_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        w_name, b_name, d_name = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
        for n in (w_name, b_name, d_name):
            check(n in sd, f"missing tensor {n}")
        w, b, d = sd[w_name], sd[b_name], sd[d_name]
        check(tuple(w.shape) == (NUM_HEADS * QKV_BLOCK, HIDDEN), f"{w_name} has shape {tuple(w.shape)}")
        check(tuple(b.shape) == (NUM_HEADS * QKV_BLOCK,), f"{b_name} has shape {tuple(b.shape)}")
        check(tuple(d.shape) == (HIDDEN, HIDDEN), f"{d_name} has shape {tuple(d.shape)}")

        qkv_start = PRUNE_HEAD * QKV_BLOCK  # 3840
        dense_start = PRUNE_HEAD * HEAD_DIM  # 1280
        out[w_name] = drop_block(w, 0, qkv_start, QKV_BLOCK)
        out[b_name] = drop_block(b, 0, qkv_start, QKV_BLOCK)
        out[d_name] = drop_block(d, 1, dense_start, HEAD_DIM)

        # Layer-level shape and dtype checks.
        check(tuple(out[w_name].shape) == (5376, HIDDEN), f"{w_name} pruned shape {tuple(out[w_name].shape)}")
        check(tuple(out[b_name].shape) == (5376,), f"{b_name} pruned shape {tuple(out[b_name].shape)}")
        check(tuple(out[d_name].shape) == (HIDDEN, 1792), f"{d_name} pruned shape {tuple(out[d_name].shape)}")
        for n in (w_name, b_name, d_name):
            check(out[n].dtype == sd[n].dtype, f"{n} dtype changed to {out[n].dtype}")
        # Value-level spot checks: kept pieces are bit-identical to source slices.
        check(torch.equal(out[w_name][:3840], w[:3840]) and torch.equal(out[w_name][3840:], w[4608:]), f"{w_name} content mismatch")
        check(torch.equal(out[b_name][:3840], b[:3840]) and torch.equal(out[b_name][3840:], b[4608:]), f"{b_name} content mismatch")
        check(torch.equal(out[d_name][:, :1280], d[:, :1280]) and torch.equal(out[d_name][:, 1280:], d[:, 1536:]), f"{d_name} content mismatch")

    # Required checks from TASK.md.
    check(tuple(out["gpt_neox.layers.0.attention.query_key_value.weight"].shape) == (5376, 2048), "layer 0 qkv weight shape")
    check(tuple(out["gpt_neox.layers.0.attention.query_key_value.bias"].shape) == (5376,), "layer 0 qkv bias shape")
    check(tuple(out["gpt_neox.layers.0.attention.dense.weight"].shape) == (2048, 1792), "layer 0 dense weight shape")
    check(len(out) == 244, f"output has {len(out)} tensors, expected 244")
    check(set(out) == set(sd), "output key set differs from input")

    # Every non-head tensor must be untouched (same object, same shape/dtype).
    touched = {f"gpt_neox.layers.{i}.attention.{s}" for i in range(NUM_LAYERS)
               for s in ("query_key_value.weight", "query_key_value.bias", "dense.weight")}
    for n, t in out.items():
        if n not in touched:
            check(t is sd[n], f"non-head tensor {n} was modified")

    check(not os.path.exists(DST), f"destination already exists: {DST}")
    save_file(out, DST, metadata={"format": "pt"})

    # Post-write verification.
    back = load_file(DST)
    check(len(back) == 244, f"written file has {len(back)} tensors")
    for n, t in out.items():
        check(tuple(back[n].shape) == tuple(t.shape) and back[n].dtype == t.dtype, f"written {n} mismatch")
    print(f"OK: wrote {DST} with {len(back)} tensors; pruned head {PRUNE_HEAD} in {NUM_LAYERS} layers")


if __name__ == "__main__":
    main()
