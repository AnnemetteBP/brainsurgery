"""T2: structured attention-head pruning of GPT-2 (124M).

Removes head 5 from every layer at the checkpoint level. GPT-2 uses Conv1D
weights laid out as [in, out], so the head axis is the *column* axis of the
fused c_attn projection and the *row* axis of c_proj.
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE = 5


def keep_within_segment() -> list[int]:
    """Column indices to keep inside one 768-wide q/k/v segment."""
    return [i for i in range(HIDDEN) if not (PRUNE * HEAD_DIM <= i < (PRUNE + 1) * HEAD_DIM)]


def main() -> None:
    if not SRC.is_file():
        raise SystemExit(f"input checkpoint not found: {SRC}")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(SRC), framework="pt") as f:
        metadata = f.metadata() or {}
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

    n_in = len(tensors)
    if n_in != 160:
        raise SystemExit(f"expected 160 input tensors, got {n_in}")

    seg = keep_within_segment()
    # fused [q | k | v]: same head block dropped in each of the three segments
    attn_cols = torch.tensor(
        [s * HIDDEN + i for s in range(3) for i in seg], dtype=torch.long
    )
    proj_rows = torch.tensor(seg, dtype=torch.long)

    for i in range(N_LAYERS):
        w_name = f"h.{i}.attn.c_attn.weight"
        b_name = f"h.{i}.attn.c_attn.bias"
        p_name = f"h.{i}.attn.c_proj.weight"
        for name, want in ((w_name, (HIDDEN, 3 * HIDDEN)), (b_name, (3 * HIDDEN,)),
                           (p_name, (HIDDEN, HIDDEN))):
            if name not in tensors:
                raise SystemExit(f"missing expected tensor: {name}")
            got = tuple(tensors[name].shape)
            if got != want:
                raise SystemExit(f"{name}: expected shape {want}, got {got}")

        tensors[w_name] = tensors[w_name].index_select(1, attn_cols).contiguous()
        tensors[b_name] = tensors[b_name].index_select(0, attn_cols).contiguous()
        tensors[p_name] = tensors[p_name].index_select(0, proj_rows).contiguous()

    # Required checks: fail loudly before writing anything.
    kept = HIDDEN - HEAD_DIM  # 704
    checks = {
        "h.0.attn.c_attn.weight": (HIDDEN, 3 * kept),
        "h.0.attn.c_attn.bias": (3 * kept,),
        "h.0.attn.c_proj.weight": (kept, HIDDEN),
    }
    for name, want in checks.items():
        got = tuple(tensors[name].shape)
        if got != want:
            raise SystemExit(f"CHECK FAILED: {name} has shape {got}, expected {want}")
    if len(tensors) != 160:
        raise SystemExit(f"CHECK FAILED: output has {len(tensors)} tensors, expected 160")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(DST), metadata=metadata or None)

    with safe_open(str(DST), framework="pt") as f:
        names = list(f.keys())
        if len(names) != 160:
            raise SystemExit(f"CHECK FAILED: written file has {len(names)} tensors")
        for name, want in checks.items():
            got = tuple(f.get_slice(name).get_shape())
            if got != want:
                raise SystemExit(f"CHECK FAILED (readback): {name} shape {got} != {want}")

    print(f"wrote {DST} with {len(names)} tensors")


if __name__ == "__main__":
    main()
