#!/usr/bin/env python3
"""T4: task-vector merge of two Pythia-1B fine-tunes into the base.

out[X] = base[X] + lam * (ft1[X] - base[X]) + lam * (ft2[X] - base[X])
for the 64 MLP tensors only; every other tensor is copied from the base.
Both task vectors are taken against the *unmodified* base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_LAYERS = 16
N_TENSORS = 244
N_MLP = 64

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # sandbox root
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = HERE / "model.safetensors"


class CheckFailed(RuntimeError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def mlp_names() -> set[str]:
    names = set()
    for i in range(N_LAYERS):
        for leaf in (
            "mlp.dense_h_to_4h.weight",
            "mlp.dense_h_to_4h.bias",
            "mlp.dense_4h_to_h.weight",
            "mlp.dense_4h_to_h.bias",
        ):
            names.add(f"gpt_neox.layers.{i}.{leaf}")
    return names


def raw_bytes(t: torch.Tensor) -> torch.Tensor:
    """Flat uint8 view, so the comparison is bit-exact (NaN-safe)."""
    return t.contiguous().reshape(-1).view(torch.uint8)


def identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(raw_bytes(a), raw_bytes(b)))


def main() -> None:
    for p in (BASE, FT1, FT2):
        check(p.is_file(), f"missing input checkpoint: {p}")

    with (
        safe_open(BASE, framework="pt", device="cpu") as fb,
        safe_open(FT1, framework="pt", device="cpu") as f1,
        safe_open(FT2, framework="pt", device="cpu") as f2,
    ):
        kb, k1, k2 = set(fb.keys()), set(f1.keys()), set(f2.keys())

        # --- step 1: same tensor names in all three -------------------------
        check(
            kb == k1,
            f"base/ft1 key sets differ: base-only={sorted(kb - k1)[:5]} "
            f"ft1-only={sorted(k1 - kb)[:5]}",
        )
        check(
            kb == k2,
            f"base/ft2 key sets differ: base-only={sorted(kb - k2)[:5]} "
            f"ft2-only={sorted(k2 - kb)[:5]}",
        )
        check(
            len(kb) == N_TENSORS,
            f"expected {N_TENSORS} tensors in the base, found {len(kb)}",
        )

        mlp = mlp_names()
        check(len(mlp) == N_MLP, f"built {len(mlp)} MLP names, expected {N_MLP}")
        missing = sorted(mlp - kb)
        check(not missing, f"MLP tensors absent from the checkpoints: {missing[:5]}")

        # --- step 1 (cont.): every non-MLP tensor identical in all three ----
        others = sorted(kb - mlp)
        check(
            len(others) == N_TENSORS - N_MLP,
            f"expected {N_TENSORS - N_MLP} shared tensors, found {len(others)}",
        )
        for name in others:
            tb = fb.get_tensor(name)
            check(
                identical(tb, f1.get_tensor(name)),
                f"shared tensor differs between base and ft1: {name}",
            )
            check(
                identical(tb, f2.get_tensor(name)),
                f"shared tensor differs between base and ft2: {name}",
            )
        print(f"[ok] step 1: {len(others)} non-MLP tensors identical in all three checkpoints")

        # --- step 2/3: merge ------------------------------------------------
        out: dict[str, torch.Tensor] = {}
        merged = 0
        for name in sorted(kb):
            tb = fb.get_tensor(name)
            if name not in mlp:
                out[name] = tb.clone()
                continue
            t1 = f1.get_tensor(name)
            t2 = f2.get_tensor(name)
            check(
                t1.shape == tb.shape and t2.shape == tb.shape,
                f"shape mismatch on {name}: base={tuple(tb.shape)} "
                f"ft1={tuple(t1.shape)} ft2={tuple(t2.shape)}",
            )
            check(
                t1.dtype == tb.dtype and t2.dtype == tb.dtype,
                f"dtype mismatch on {name}: base={tb.dtype} ft1={t1.dtype} ft2={t2.dtype}",
            )
            b32 = tb.to(torch.float32)
            acc = b32 + LAMBDA * (t1.to(torch.float32) - b32) + LAMBDA * (t2.to(torch.float32) - b32)
            res = acc.to(tb.dtype)
            check(torch.isfinite(acc).all().item(), f"non-finite value produced for {name}")
            out[name] = res.contiguous()
            merged += 1

        metadata = fb.metadata() or {}

    # --- required checks ---------------------------------------------------
    check(merged == N_MLP, f"merged {merged} tensors, expected exactly {N_MLP}")
    check(len(out) == N_TENSORS, f"output has {len(out)} tensors, expected {N_TENSORS}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata=metadata or None)

    # --- read back and verify what landed on disk --------------------------
    with safe_open(OUT, framework="pt", device="cpu") as fo:
        ko = set(fo.keys())
        check(ko == set(out.keys()), "written key set does not match the merge result")
        check(len(ko) == N_TENSORS, f"written file has {len(ko)} tensors, expected {N_TENSORS}")
        with safe_open(BASE, framework="pt", device="cpu") as fb:
            for name in sorted(ko):
                to = fo.get_tensor(name)
                tb = fb.get_tensor(name)
                check(
                    to.shape == tb.shape and to.dtype == tb.dtype,
                    f"written {name}: shape/dtype drifted from the base",
                )
                if name not in mlp_names():
                    check(identical(to, tb), f"unchanged tensor not bit-exact on disk: {name}")

    print(f"[ok] merged {merged} MLP tensors with lambda={LAMBDA}")
    print(f"[ok] wrote {OUT} with {len(out)} tensors")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
