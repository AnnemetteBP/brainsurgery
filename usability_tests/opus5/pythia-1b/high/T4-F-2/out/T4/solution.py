#!/usr/bin/env python
"""T4: task-vector merge of two Pythia-1B fine-tunes onto the base.

    out[X] = base[X] + lam * (ft1[X] - base[X]) + lam * (ft2[X] - base[X])

for the 64 MLP tensors only; every other tensor is copied bit-exactly from the
base. Both task vectors are taken against the *unmodified* base.

The script aborts (non-zero exit) if any required check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUT = Path("out/T4/model.safetensors")

LAMBDA = 0.4
N_LAYERS = 16
N_EXPECTED_MERGED = 64
N_EXPECTED_TOTAL = 244

MLP_KEYS = [
    f"gpt_neox.layers.{i}.mlp.{proj}.{kind}"
    for i in range(N_LAYERS)
    for proj in ("dense_h_to_4h", "dense_4h_to_h")
    for kind in ("weight", "bias")
]


class CheckFailed(RuntimeError):
    """A required check did not hold."""


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def bit_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact comparison, dtype- and NaN-agnostic (compares raw bytes)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    # reshape(-1) first: a 0-dim tensor cannot be re-viewed as a wider dtype
    ab = a.contiguous().reshape(-1).view(torch.uint8)
    bb = b.contiguous().reshape(-1).view(torch.uint8)
    return not bool(ab.ne(bb).any())


def main() -> None:
    with (
        safe_open(BASE, framework="pt") as base,
        safe_open(FT1, framework="pt") as ft1,
        safe_open(FT2, framework="pt") as ft2,
    ):
        # --- step 1: the three checkpoints must agree everywhere but the MLPs ---
        kb, k1, k2 = set(base.keys()), set(ft1.keys()), set(ft2.keys())
        check(kb == k1, f"ft1 key set differs from base: {sorted(kb ^ k1)[:5]}")
        check(kb == k2, f"ft2 key set differs from base: {sorted(kb ^ k2)[:5]}")
        check(
            len(kb) == N_EXPECTED_TOTAL,
            f"expected {N_EXPECTED_TOTAL} tensors in the base, found {len(kb)}",
        )

        mlp = set(MLP_KEYS)
        check(
            len(mlp) == N_EXPECTED_MERGED,
            f"MLP name list has {len(mlp)} unique names, expected {N_EXPECTED_MERGED}",
        )
        missing = sorted(mlp - kb)
        check(not missing, f"MLP tensors absent from the checkpoints: {missing[:5]}")

        shared = sorted(kb - mlp)
        check(
            len(shared) == N_EXPECTED_TOTAL - N_EXPECTED_MERGED,
            f"expected {N_EXPECTED_TOTAL - N_EXPECTED_MERGED} shared tensors,"
            f" found {len(shared)}",
        )

        out: dict[str, torch.Tensor] = {}
        differing: list[str] = []
        for name in shared:
            tb = base.get_tensor(name)
            if not bit_identical(tb, ft1.get_tensor(name)):
                differing.append(f"{name} (ft1)")
            if not bit_identical(tb, ft2.get_tensor(name)):
                differing.append(f"{name} (ft2)")
            out[name] = tb
        check(
            not differing,
            f"{len(differing)} non-MLP tensor(s) differ from the base, the frozen-"
            f"backbone precondition does not hold: {differing[:5]}",
        )

        # --- step 2: merge the MLP tensors, both deltas against the pristine base ---
        merged = 0
        for name in MLP_KEYS:
            tb = base.get_tensor(name)
            t1 = ft1.get_tensor(name)
            t2 = ft2.get_tensor(name)
            check(
                t1.shape == tb.shape == t2.shape,
                f"shape mismatch on {name}: {tb.shape} / {t1.shape} / {t2.shape}",
            )
            check(
                t1.dtype == tb.dtype == t2.dtype,
                f"dtype mismatch on {name}: {tb.dtype} / {t1.dtype} / {t2.dtype}",
            )
            b32 = tb.to(torch.float32)
            acc = b32 + LAMBDA * (t1.to(torch.float32) - b32)
            acc += LAMBDA * (t2.to(torch.float32) - b32)
            out[name] = acc.to(tb.dtype).contiguous()
            merged += 1

    check(
        merged == N_EXPECTED_MERGED,
        f"merged {merged} tensors, expected exactly {N_EXPECTED_MERGED}",
    )
    check(
        len(out) == N_EXPECTED_TOTAL,
        f"output holds {len(out)} tensors, expected exactly {N_EXPECTED_TOTAL}",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT, metadata={"format": "pt"})

    # --- post-write verification against what actually landed on disk ---
    with safe_open(OUT, framework="pt") as w:
        keys = set(w.keys())
        check(
            len(keys) == N_EXPECTED_TOTAL,
            f"written file holds {len(keys)} tensors, expected {N_EXPECTED_TOTAL}",
        )
        check(keys == kb, "written key set differs from the base key set")
        changed = sum(
            0 if bit_identical(w.get_tensor(n), out[n]) else 1 for n in sorted(keys)
        )
        check(changed == 0, f"{changed} tensor(s) did not round-trip through the file")

    print(f"merged {merged} MLP tensors, copied {len(shared)} tensors from the base")
    print(f"wrote {OUT} with {len(out)} tensors")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as exc:
        sys.exit(f"CHECK FAILED: {exc}")
