"""T2: structured attention-head pruning for GPT-2 (124M).

Removes head 5 from every layer at the checkpoint level. GPT-2 uses Conv1D
weights stored as [in, out], so for the fused c_attn projection the heads are
*column* blocks inside each of the three 768-wide [q | k | v] segments, while
for the c_proj output projection the heads are *row* blocks.
"""

from __future__ import annotations

import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T2"
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
N_SEGMENTS = 3  # q, k, v fused in c_attn
PRUNE_HEAD = 5


def keep_columns_in_segment() -> list[int]:
    """Column offsets to keep inside one 768-wide q/k/v segment."""
    keep: list[int] = []
    for head in range(N_HEADS):
        if head == PRUNE_HEAD:
            continue
        keep.extend(range(head * HEAD_DIM, (head + 1) * HEAD_DIM))
    return keep


def keep_columns_fused() -> list[int]:
    """Column indices to keep across the fused [q | k | v] c_attn tensor."""
    within = keep_columns_in_segment()
    keep: list[int] = []
    for seg in range(N_SEGMENTS):
        keep.extend(seg * HIDDEN + c for c in within)
    return keep


def ranges_to_index(ranges: list[tuple[int, int]]) -> list[int]:
    """Inclusive [lo, hi] ranges -> flat index list, in the order given."""
    out: list[int] = []
    for lo, hi in ranges:
        out.extend(range(lo, hi + 1))
    return out


# The ranges the task statement spells out, used to cross-check the derivation.
SPEC_FUSED_RANGES = [
    (0, 319), (384, 767),
    (768, 1087), (1152, 1535),
    (1536, 1855), (1920, 2303),
]
SPEC_PROJ_RANGES = [(0, 319), (384, 767)]


def main() -> None:
    fused_keep = keep_columns_fused()
    proj_keep = keep_columns_in_segment()

    # Cross-check the head-arithmetic against the literal ranges in the task.
    if fused_keep != ranges_to_index(SPEC_FUSED_RANGES):
        raise AssertionError("derived c_attn keep index disagrees with the task ranges")
    if proj_keep != ranges_to_index(SPEC_PROJ_RANGES):
        raise AssertionError("derived c_proj keep index disagrees with the task ranges")
    if len(fused_keep) != 2112 or len(proj_keep) != 704:
        raise AssertionError(
            f"unexpected keep sizes: c_attn={len(fused_keep)}, c_proj={len(proj_keep)}"
        )

    fused_idx = torch.tensor(fused_keep, dtype=torch.long)
    proj_idx = torch.tensor(proj_keep, dtype=torch.long)

    with safe_open(IN_PATH, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}

    if len(tensors) != 160:
        raise AssertionError(f"input has {len(tensors)} tensors, expected 160")

    expected_in = {
        "attn.c_attn.weight": (HIDDEN, N_SEGMENTS * HIDDEN),
        "attn.c_attn.bias": (N_SEGMENTS * HIDDEN,),
        "attn.c_proj.weight": (HIDDEN, HIDDEN),
    }
    expected_out = {
        "attn.c_attn.weight": (HIDDEN, 2112),
        "attn.c_attn.bias": (2112,),
        "attn.c_proj.weight": (704, HIDDEN),
    }

    touched = 0
    for i in range(N_LAYERS):
        for suffix, exp_shape in expected_in.items():
            name = f"h.{i}.{suffix}"
            if name not in tensors:
                raise AssertionError(f"missing expected tensor {name}")
            if tuple(tensors[name].shape) != exp_shape:
                raise AssertionError(
                    f"{name} has shape {tuple(tensors[name].shape)}, expected {exp_shape}"
                )

        w = tensors[f"h.{i}.attn.c_attn.weight"]
        b = tensors[f"h.{i}.attn.c_attn.bias"]
        p = tensors[f"h.{i}.attn.c_proj.weight"]

        # c_attn: heads live along the columns (dim 1) of the [in, out] weight,
        # and along the single axis of the matching bias.
        tensors[f"h.{i}.attn.c_attn.weight"] = w.index_select(1, fused_idx).contiguous()
        tensors[f"h.{i}.attn.c_attn.bias"] = b.index_select(0, fused_idx).contiguous()
        # c_proj: heads live along the rows (dim 0), the input side of [in, out].
        tensors[f"h.{i}.attn.c_proj.weight"] = p.index_select(0, proj_idx).contiguous()
        touched += 3

    if touched != N_LAYERS * 3:
        raise AssertionError(f"edited {touched} tensors, expected {N_LAYERS * 3}")

    # Required checks, before writing anything.
    for i in range(N_LAYERS):
        for suffix, exp_shape in expected_out.items():
            name = f"h.{i}.{suffix}"
            got = tuple(tensors[name].shape)
            if got != exp_shape:
                raise AssertionError(f"{name} has shape {got}, expected {exp_shape}")
    if len(tensors) != 160:
        raise AssertionError(f"output would have {len(tensors)} tensors, expected 160")
    if any(t.dtype != torch.float32 for t in tensors.values()):
        raise AssertionError("output contains a non-float32 tensor")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(tensors, OUT_PATH, metadata=metadata)

    # Read the file back and re-assert the required checks on what landed on disk.
    with safe_open(OUT_PATH, framework="pt", device="cpu") as f:
        written = {k: f.get_slice(k).get_shape() for k in f.keys()}
    if len(written) != 160:
        raise AssertionError(f"written file has {len(written)} tensors, expected 160")
    for name, exp_shape in (
        ("h.0.attn.c_attn.weight", [768, 2112]),
        ("h.0.attn.c_attn.bias", [2112]),
        ("h.0.attn.c_proj.weight", [704, 768]),
    ):
        if written[name] != exp_shape:
            raise AssertionError(f"{name} on disk has shape {written[name]}, expected {exp_shape}")

    print(f"wrote {OUT_PATH}: {len(written)} tensors, pruned head {PRUNE_HEAD} in {N_LAYERS} layers")
    print(f"  h.0.attn.c_attn.weight {written['h.0.attn.c_attn.weight']}")
    print(f"  h.0.attn.c_attn.bias   {written['h.0.attn.c_attn.bias']}")
    print(f"  h.0.attn.c_proj.weight {written['h.0.attn.c_proj.weight']}")


if __name__ == "__main__":
    main()
