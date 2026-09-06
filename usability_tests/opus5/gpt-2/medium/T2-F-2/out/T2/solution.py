"""T2: remove attention head 5 from every layer of GPT-2 (124M).

Plain torch + safetensors script. GPT-2 uses Conv1D, so projection weights are
stored as [in, out]:
  attn.c_attn.weight [768, 2304] -> columns, three 768-wide segments q|k|v,
                                    12 head blocks of 64 columns per segment
  attn.c_attn.bias   [2304]      -> same layout along dim 0
  attn.c_proj.weight [768, 768]  -> rows are the head blocks (input side)
Everything else, including attn.c_proj.bias and the attn.bias mask buffer,
is copied through untouched.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE = 5


def keep_indices(n_segments: int) -> torch.Tensor:
    """Indices to keep along a head-blocked axis of n_segments * 768 entries."""
    keep = [
        s * HIDDEN + h * HEAD_DIM + d
        for s in range(n_segments)
        for h in range(N_HEADS)
        if h != PRUNE
        for d in range(HEAD_DIM)
    ]
    return torch.tensor(keep, dtype=torch.long)


QKV_KEEP = keep_indices(3)  # 2112 columns of c_attn
OUT_KEEP = keep_indices(1)  # 704 rows of c_proj


def main() -> None:
    src = load_file(str(SRC))
    n_in = len(src)

    out: dict[str, torch.Tensor] = {}
    touched = 0
    for name, t in src.items():
        if name.endswith(".attn.c_attn.weight"):
            assert t.shape == (HIDDEN, 3 * HIDDEN), f"{name}: {tuple(t.shape)}"
            new = t.index_select(1, QKV_KEEP)
        elif name.endswith(".attn.c_attn.bias"):
            assert t.shape == (3 * HIDDEN,), f"{name}: {tuple(t.shape)}"
            new = t.index_select(0, QKV_KEEP)
        elif name.endswith(".attn.c_proj.weight"):
            assert t.shape == (HIDDEN, HIDDEN), f"{name}: {tuple(t.shape)}"
            new = t.index_select(0, OUT_KEEP)
        else:
            new = t
            out[name] = new
            continue
        touched += 1
        out[name] = new.contiguous().clone()

    # --- required checks: fail loudly before writing -------------------------
    assert touched == 3 * N_LAYERS, f"touched {touched} head-bearing tensors, want 36"
    assert len(out) == n_in == 160, f"tensor count {len(out)} (input {n_in}), want 160"
    checks = {
        "h.0.attn.c_attn.weight": (768, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, 768),
    }
    for k, want in checks.items():
        got = tuple(out[k].shape)
        assert got == want, f"{k} has shape {got}, want {want}"
    for i in range(N_LAYERS):
        assert tuple(out[f"h.{i}.attn.c_attn.weight"].shape) == (768, 2112)
        assert tuple(out[f"h.{i}.attn.c_attn.bias"].shape) == (2112,)
        assert tuple(out[f"h.{i}.attn.c_proj.weight"].shape) == (704, 768)
        assert tuple(out[f"h.{i}.attn.c_proj.bias"].shape) == (768,)
        assert tuple(out[f"h.{i}.attn.bias"].shape) == (1, 1, 1024, 1024)
        assert tuple(out[f"h.{i}.mlp.c_proj.weight"].shape) == (3072, 768)
    assert set(out) == set(src), "key set changed"
    for k in out:
        assert out[k].dtype == src[k].dtype, f"{k}: dtype changed"

    # spot-check the block boundary: kept column 320 must be old column 384
    w = src["h.0.attn.c_attn.weight"]
    assert torch.equal(out["h.0.attn.c_attn.weight"][:, 320], w[:, 384])
    assert torch.equal(out["h.0.attn.c_attn.weight"][:, 319], w[:, 319])
    assert torch.equal(out["h.0.attn.c_attn.weight"][:, 704], w[:, 768])
    assert torch.equal(out["h.0.attn.c_proj.weight"][320], src["h.0.attn.c_proj.weight"][384])

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata={"format": "pt"})

    back = load_file(str(DST))
    assert len(back) == 160, f"wrote {len(back)} tensors"
    assert tuple(back["h.0.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(back["h.0.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(back["h.0.attn.c_proj.weight"].shape) == (704, 768)
    print(f"wrote {DST} with {len(back)} tensors; pruned head {PRUNE} in {N_LAYERS} layers")


if __name__ == "__main__":
    main()
