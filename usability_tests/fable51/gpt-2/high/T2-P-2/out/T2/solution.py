"""T2: remove attention head 5 from every layer of GPT-2 (124M) at the checkpoint level."""

import os
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5
EXPECTED_TENSORS = 160


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def keep_index_for_segment() -> torch.Tensor:
    """Indices within one 768-wide segment that survive after dropping head PRUNE_HEAD."""
    idx = torch.arange(HIDDEN)
    lo, hi = PRUNE_HEAD * HEAD_DIM, (PRUNE_HEAD + 1) * HEAD_DIM
    return torch.cat([idx[:lo], idx[hi:]])


def main() -> None:
    sd = load_file(SRC)
    if len(sd) != EXPECTED_TENSORS:
        fail(f"input has {len(sd)} tensors, expected {EXPECTED_TENSORS}")

    seg_keep = keep_index_for_segment()  # 704 entries: 0..319, 384..767
    # fused [q | k | v]: same pattern in each of the three 768-wide segments
    qkv_keep = torch.cat([seg_keep + s * HIDDEN for s in range(3)])  # 2112 entries

    out = {}
    for name, t in sd.items():
        out[name] = t

    for i in range(N_LAYERS):
        w_name = f"h.{i}.attn.c_attn.weight"
        b_name = f"h.{i}.attn.c_attn.bias"
        p_name = f"h.{i}.attn.c_proj.weight"
        for n in (w_name, b_name, p_name):
            if n not in sd:
                fail(f"missing tensor {n}")

        w = sd[w_name]
        b = sd[b_name]
        p = sd[p_name]
        if tuple(w.shape) != (HIDDEN, 3 * HIDDEN):
            fail(f"{w_name} has shape {tuple(w.shape)}, expected {(HIDDEN, 3 * HIDDEN)}")
        if tuple(b.shape) != (3 * HIDDEN,):
            fail(f"{b_name} has shape {tuple(b.shape)}, expected {(3 * HIDDEN,)}")
        if tuple(p.shape) != (HIDDEN, HIDDEN):
            fail(f"{p_name} has shape {tuple(p.shape)}, expected {(HIDDEN, HIDDEN)}")

        # Conv1D layout [in, out]: heads are column blocks of c_attn, row blocks of c_proj.
        out[w_name] = w.index_select(1, qkv_keep).contiguous()
        out[b_name] = b.index_select(0, qkv_keep).contiguous()
        out[p_name] = p.index_select(0, seg_keep).contiguous()

    # Required checks.
    checks = {
        "h.0.attn.c_attn.weight": (HIDDEN, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, HIDDEN),
    }
    for n, shape in checks.items():
        if tuple(out[n].shape) != shape:
            fail(f"{n} has shape {tuple(out[n].shape)}, expected {shape}")
    if len(out) != EXPECTED_TENSORS:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    # Extra sanity: every layer got the same treatment; untouched tensors are identical.
    for i in range(N_LAYERS):
        if tuple(out[f"h.{i}.attn.c_attn.weight"].shape) != (HIDDEN, 2112):
            fail(f"layer {i} c_attn.weight has wrong shape")
        if tuple(out[f"h.{i}.attn.c_attn.bias"].shape) != (2112,):
            fail(f"layer {i} c_attn.bias has wrong shape")
        if tuple(out[f"h.{i}.attn.c_proj.weight"].shape) != (704, HIDDEN):
            fail(f"layer {i} c_proj.weight has wrong shape")
    for n, t in sd.items():
        if out[n].dtype != t.dtype:
            fail(f"{n} dtype changed")
        if n.endswith(("attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight")) and ".attn." in n:
            continue
        if out[n].shape != t.shape or not torch.equal(out[n], t):
            fail(f"{n} should be unchanged")

    if os.path.exists(DST):
        fail(f"destination already exists: {DST}")
    os.makedirs(HERE, exist_ok=True)
    save_file(out, DST, metadata={"format": "pt"})

    # Verify the written file.
    back = load_file(DST)
    if len(back) != EXPECTED_TENSORS:
        fail(f"written file has {len(back)} tensors")
    for n, shape in checks.items():
        if tuple(back[n].shape) != shape:
            fail(f"written {n} has shape {tuple(back[n].shape)}")
    print(f"OK: wrote {DST} with {len(back)} tensors")


if __name__ == "__main__":
    main()
