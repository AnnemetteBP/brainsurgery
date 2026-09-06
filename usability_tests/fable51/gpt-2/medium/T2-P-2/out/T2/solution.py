"""T2: prune attention head 5 from every layer of GPT-2 (124M)."""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 12, 12, 64, 768
PRUNE_HEAD = 5


def keep_index(n_segments: int) -> torch.Tensor:
    """Indices to keep along a head-blocked axis made of n_segments 768-wide segments."""
    idx = []
    for s in range(n_segments):
        base = s * HIDDEN
        for h in range(N_HEADS):
            if h == PRUNE_HEAD:
                continue
            idx.extend(range(base + h * HEAD_DIM, base + (h + 1) * HEAD_DIM))
    return torch.tensor(idx, dtype=torch.long)


def main() -> None:
    sd = load_file(SRC)
    if len(sd) != 160:
        raise SystemExit(f"expected 160 input tensors, got {len(sd)}")

    qkv_idx = keep_index(3)  # [q | k | v], each 768 wide
    proj_idx = keep_index(1)
    assert qkv_idx.numel() == 2112 and proj_idx.numel() == 704

    out = {}
    for name, t in sd.items():
        out[name] = t
    for i in range(N_LAYERS):
        w = f"h.{i}.attn.c_attn.weight"
        b = f"h.{i}.attn.c_attn.bias"
        p = f"h.{i}.attn.c_proj.weight"
        for k, shape in ((w, (768, 2304)), (b, (2304,)), (p, (768, 768))):
            if k not in sd:
                raise SystemExit(f"missing tensor {k}")
            if tuple(sd[k].shape) != shape:
                raise SystemExit(f"{k}: expected shape {shape}, got {tuple(sd[k].shape)}")
        out[w] = sd[w].index_select(1, qkv_idx).contiguous()
        out[b] = sd[b].index_select(0, qkv_idx).contiguous()
        out[p] = sd[p].index_select(0, proj_idx).contiguous()

    # Required checks.
    checks = {
        "h.0.attn.c_attn.weight": (768, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, 768),
    }
    for k, shape in checks.items():
        if tuple(out[k].shape) != shape:
            raise SystemExit(f"CHECK FAILED: {k} has shape {tuple(out[k].shape)}, want {shape}")
    if len(out) != 160:
        raise SystemExit(f"CHECK FAILED: output has {len(out)} tensors, want 160")
    # Spot-check block order on layer 0 against explicit slices.
    w0 = sd["h.0.attn.c_attn.weight"]
    ref = torch.cat([w0[:, 0:320], w0[:, 384:768], w0[:, 768:1088], w0[:, 1152:1536],
                     w0[:, 1536:1856], w0[:, 1920:2304]], dim=1)
    if not torch.equal(ref, out["h.0.attn.c_attn.weight"]):
        raise SystemExit("CHECK FAILED: c_attn column order mismatch")
    p0 = sd["h.0.attn.c_proj.weight"]
    if not torch.equal(torch.cat([p0[0:320], p0[384:768]], 0), out["h.0.attn.c_proj.weight"]):
        raise SystemExit("CHECK FAILED: c_proj row order mismatch")
    for k in sd:
        if k not in out or out[k].dtype != sd[k].dtype:
            raise SystemExit(f"CHECK FAILED: {k} missing or dtype changed")

    if os.path.exists(DST):
        raise SystemExit(f"destination already exists: {DST}")
    save_file(out, DST)
    print(f"wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    sys.exit(main())
