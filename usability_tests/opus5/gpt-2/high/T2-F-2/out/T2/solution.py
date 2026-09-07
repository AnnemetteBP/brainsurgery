"""T2: structured attention-head pruning for GPT-2 (124M).

Removes head 5 from every layer at the checkpoint level, slicing the fused
[q|k|v] input projection (column blocks) and the output projection (row
blocks). Conv1D layout: GPT-2 stores projections as [in, out].
"""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5


def keep_within_segment() -> list[int]:
    """Column/row indices to keep inside one 768-wide projection segment."""
    return [i for i in range(HIDDEN) if i // HEAD_DIM != PRUNE_HEAD]


def ranges_to_indices(ranges: list[tuple[int, int]]) -> list[int]:
    """Inclusive [start, end] ranges -> flat index list, in the given order."""
    out: list[int] = []
    for start, end in ranges:
        out.extend(range(start, end + 1))
    return out


# Indices derived from head geometry.
KEEP_SEG = keep_within_segment()                                   # 704 per segment
KEEP_QKV = [q * HIDDEN + i for q in range(3) for i in KEEP_SEG]     # 2112 columns
KEEP_OUT = list(KEEP_SEG)                                           # 704 rows

# The same indices spelled out by TASK.md; cross-check the derivation.
SPEC_QKV = ranges_to_indices(
    [(0, 319), (384, 767), (768, 1087), (1152, 1535), (1536, 1855), (1920, 2303)]
)
SPEC_OUT = ranges_to_indices([(0, 319), (384, 767)])
assert KEEP_QKV == SPEC_QKV, "derived q|k|v indices disagree with the task spec"
assert KEEP_OUT == SPEC_OUT, "derived output-projection indices disagree with the task spec"


def main() -> None:
    qkv_idx = torch.tensor(KEEP_QKV, dtype=torch.long)
    out_idx = torch.tensor(KEEP_OUT, dtype=torch.long)

    touched: set[str] = set()
    for i in range(N_LAYERS):
        touched.update(
            {
                f"h.{i}.attn.c_attn.weight",
                f"h.{i}.attn.c_attn.bias",
                f"h.{i}.attn.c_proj.weight",
            }
        )

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(SRC, framework="pt") as f:
        src_keys = list(f.keys())
        missing = touched - set(src_keys)
        if missing:
            raise KeyError(f"expected head-bearing tensors missing from input: {sorted(missing)}")
        for key in src_keys:
            t = f.get_tensor(key)
            if key.endswith("attn.c_attn.weight"):
                if t.shape != (HIDDEN, 3 * HIDDEN):
                    raise ValueError(f"{key}: unexpected shape {tuple(t.shape)}")
                t = t.index_select(1, qkv_idx)
            elif key.endswith("attn.c_attn.bias"):
                if t.shape != (3 * HIDDEN,):
                    raise ValueError(f"{key}: unexpected shape {tuple(t.shape)}")
                t = t.index_select(0, qkv_idx)
            elif key.endswith("attn.c_proj.weight"):
                if t.shape != (HIDDEN, HIDDEN):
                    raise ValueError(f"{key}: unexpected shape {tuple(t.shape)}")
                t = t.index_select(0, out_idx)
            tensors[key] = t.contiguous()

    # Required checks: fail loudly before writing.
    expected = {
        "h.0.attn.c_attn.weight": (768, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, 768),
    }
    for key, shape in expected.items():
        got = tuple(tensors[key].shape)
        if got != shape:
            raise AssertionError(f"{key}: expected shape {shape}, got {got}")
    if len(tensors) != 160:
        raise AssertionError(f"expected exactly 160 tensors, got {len(tensors)}")

    # Same checks for every layer, and nothing else may change.
    with safe_open(SRC, framework="pt") as f:
        for key in tensors:
            src = f.get_tensor(key)
            if tensors[key].dtype != src.dtype:
                raise AssertionError(f"{key}: dtype changed {src.dtype} -> {tensors[key].dtype}")
            if key in touched:
                continue
            if tensors[key].shape != src.shape or not torch.equal(tensors[key], src):
                raise AssertionError(f"{key}: should have been left unchanged")
        for i in range(N_LAYERS):
            shapes = {
                f"h.{i}.attn.c_attn.weight": (768, 2112),
                f"h.{i}.attn.c_attn.bias": (2112,),
                f"h.{i}.attn.c_proj.weight": (704, 768),
            }
            for key, shape in shapes.items():
                got = tuple(tensors[key].shape)
                if got != shape:
                    raise AssertionError(f"{key}: expected shape {shape}, got {got}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, DST)
    print(f"wrote {DST} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
