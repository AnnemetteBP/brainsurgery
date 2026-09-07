#!/usr/bin/env python3
"""T4: task-vector merge of two fine-tunes of OLMo-1B-0724-hf.

    out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors only; every other tensor is copied from the base.
All checks are hard assertions: the script exits non-zero if any fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_TOTAL = 114
N_MLP = 48

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent            # sandbox root
BASE_DIR = ROOT / "inputs" / "base"
FT1_FILE = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_FILE = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT_FILE = HERE / "model.safetensors"

MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


class CheckError(RuntimeError):
    """A required check failed."""


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckError(msg)


class Checkpoint:
    """Read-only view over one checkpoint, sharded or single-file."""

    def __init__(self, label: str, path: Path):
        self.label = label
        if path.is_dir():
            index = path / "model.safetensors.index.json"
            require(index.is_file(), f"{label}: no index at {index}")
            weight_map = json.loads(index.read_text())["weight_map"]
            self._where = {k: path / v for k, v in weight_map.items()}
        else:
            require(path.is_file(), f"{label}: missing file {path}")
            with safe_open(path, framework="pt") as f:
                self._where = {k: path for k in f.keys()}
        self._handles: dict[Path, object] = {}

    def _handle(self, file: Path):
        h = self._handles.get(file)
        if h is None:
            h = safe_open(file, framework="pt")
            h.__enter__()
            self._handles[file] = h
        return h

    def keys(self) -> set[str]:
        return set(self._where)

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self._where[name]).get_tensor(name)

    def close(self) -> None:
        for h in self._handles.values():
            h.__exit__(None, None, None)
        self._handles.clear()


def bit_identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact comparison (unlike torch.equal, treats NaN payloads as equal)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    ia = a.contiguous().view(torch.uint8)
    ib = b.contiguous().view(torch.uint8)
    return bool(torch.equal(ia, ib))


def main() -> int:
    base = Checkpoint("base", BASE_DIR)
    ft1 = Checkpoint("ft1", FT1_FILE)
    ft2 = Checkpoint("ft2", FT2_FILE)

    # --- check 1: identical name sets, of the expected size ------------------
    kb, k1, k2 = base.keys(), ft1.keys(), ft2.keys()
    require(kb == k1, f"name sets differ base vs ft1: {sorted(kb ^ k1)[:10]}")
    require(kb == k2, f"name sets differ base vs ft2: {sorted(kb ^ k2)[:10]}")
    require(len(kb) == N_TOTAL, f"expected {N_TOTAL} tensors, found {len(kb)}")

    # --- the 48 MLP tensors, derived from the names and cross-checked --------
    mlp = sorted(k for k in kb if MLP_RE.match(k))
    layers = sorted({int(MLP_RE.match(k).group(1)) for k in mlp})
    expected = sorted(
        f"model.layers.{i}.mlp.{p}.weight"
        for i in layers
        for p in ("gate_proj", "up_proj", "down_proj")
    )
    require(mlp == expected, "MLP tensor set is not exactly 3 projections per layer")
    require(len(mlp) == N_MLP, f"expected {N_MLP} MLP tensors, found {len(mlp)}")
    mlp_set = set(mlp)
    shared = sorted(kb - mlp_set)
    require(
        len(shared) == N_TOTAL - N_MLP,
        f"expected {N_TOTAL - N_MLP} non-MLP tensors, found {len(shared)}",
    )
    print(f"[plan] {len(mlp)} MLP tensors over layers {layers[0]}..{layers[-1]}, "
          f"{len(shared)} shared tensors, lambda={LAMBDA}")

    out: dict[str, torch.Tensor] = {}

    # --- check 1 (cont.): every non-MLP tensor identical in all three --------
    # Done BEFORE any arithmetic, so we abort without touching anything.
    for name in shared:
        b, a1, a2 = base.get(name), ft1.get(name), ft2.get(name)
        require(
            b.shape == a1.shape == a2.shape,
            f"{name}: shape mismatch {tuple(b.shape)} / {tuple(a1.shape)} / {tuple(a2.shape)}",
        )
        require(
            b.dtype == a1.dtype == a2.dtype,
            f"{name}: dtype mismatch {b.dtype} / {a1.dtype} / {a2.dtype}",
        )
        require(bit_identical(b, a1), f"{name}: ft1 differs from base outside the MLP tensors")
        require(bit_identical(b, a2), f"{name}: ft2 differs from base outside the MLP tensors")
        out[name] = b
        del a1, a2
    print(f"[check] {len(shared)} non-MLP tensors verified identical in base, ft1 and ft2")

    # --- the merge ------------------------------------------------------------
    merged = 0
    for name in mlp:
        b, a1, a2 = base.get(name), ft1.get(name), ft2.get(name)
        require(
            b.shape == a1.shape == a2.shape,
            f"{name}: shape mismatch {tuple(b.shape)} / {tuple(a1.shape)} / {tuple(a2.shape)}",
        )
        require(
            b.dtype == a1.dtype == a2.dtype,
            f"{name}: dtype mismatch {b.dtype} / {a1.dtype} / {a2.dtype}",
        )
        bf = b.to(torch.float32)
        # Both task vectors are taken against the UNMODIFIED base `bf`.
        tv1 = a1.to(torch.float32) - bf
        tv2 = a2.to(torch.float32) - bf
        out[name] = (bf + LAMBDA * tv1 + LAMBDA * tv2).to(b.dtype).contiguous()
        merged += 1
        del b, a1, a2, bf, tv1, tv2

    # --- check 2 and 3 --------------------------------------------------------
    require(merged == N_MLP, f"merged {merged} tensors, expected {N_MLP}")
    require(len(out) == N_TOTAL, f"output has {len(out)} tensors, expected {N_TOTAL}")
    require(set(out) == kb, "output key set differs from the base key set")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE), metadata={"format": "pt"})
    print(f"[write] {OUT_FILE} ({OUT_FILE.stat().st_size} bytes)")
    del out

    # --- verify what actually landed on disk ---------------------------------
    with safe_open(OUT_FILE, framework="pt") as f:
        got = set(f.keys())
        require(len(got) == N_TOTAL, f"written file has {len(got)} tensors, expected {N_TOTAL}")
        require(got == kb, "written file key set differs from the base key set")
        changed = 0
        verified = 0
        for name in sorted(got):
            t = f.get_tensor(name)
            b = base.get(name)
            require(t.shape == b.shape, f"{name}: written shape {tuple(t.shape)} != {tuple(b.shape)}")
            require(t.dtype == b.dtype, f"{name}: written dtype {t.dtype} != {b.dtype}")
            if name in mlp_set:
                # recompute the merge independently and compare against disk
                bf = b.to(torch.float32)
                want = bf + LAMBDA * (ft1.get(name).to(torch.float32) - bf) \
                          + LAMBDA * (ft2.get(name).to(torch.float32) - bf)
                err = (t.to(torch.float32) - want).norm() / want.norm().clamp_min(1e-12)
                require(err <= 1e-7, f"{name}: written value off by relative {err:.3e}")
                verified += 1
                if not bit_identical(t, b):
                    changed += 1
            else:
                require(bit_identical(t, b), f"{name}: non-MLP tensor was modified on disk")
        require(
            verified == N_MLP,
            f"{verified} MLP tensors re-verified on disk, expected {N_MLP}",
        )
    print(f"[check] on disk: {N_TOTAL} tensors, {N_MLP} merged values re-verified "
          f"({changed} differ from base), "
          f"{N_TOTAL - N_MLP} shared tensors bit-identical to base")

    base.close(); ft1.close(); ft2.close()
    print("OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckError as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
