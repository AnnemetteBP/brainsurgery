"""T4: task-vector merge of two fine-tunes of OLMo-1B-0724-hf.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the 48 MLP tensors
out[X] = base[X]                                                      for everything else

Both task vectors are taken against the *unmodified* base.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
BASE_DIR = ROOT / "inputs" / "base"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
N_MLP_EXPECTED = 48
N_TOTAL_EXPECTED = 114

MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


class Check(Exception):
    """A required check failed."""


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise Check(msg)


class Shards:
    """Read-only view over one or more safetensors files, keyed by tensor name."""

    def __init__(self, paths: list[Path]) -> None:
        self._handles = [safe_open(str(p), framework="pt", device="cpu") for p in paths]
        self._owner: dict[str, object] = {}
        for h in self._handles:
            for k in h.keys():
                require(k not in self._owner, f"duplicate tensor name across shards: {k}")
                self._owner[k] = h

    def keys(self) -> set[str]:
        return set(self._owner)

    def get(self, name: str) -> torch.Tensor:
        return self._owner[name].get_tensor(name)


def open_base() -> Shards:
    index = BASE_DIR / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted({BASE_DIR / f for f in weight_map.values()})
    else:
        files = sorted(BASE_DIR.glob("*.safetensors"))
    require(bool(files), f"no safetensors shards found under {BASE_DIR}")
    return Shards(list(files))


def main() -> int:
    base = open_base()
    ft1 = Shards([FT1])
    ft2 = Shards([FT2])

    # --- Step 1: verification, before touching anything ---------------------
    kb, k1, k2 = base.keys(), ft1.keys(), ft2.keys()
    require(
        kb == k1,
        f"base/ft1 tensor names differ: only-base={sorted(kb - k1)[:5]} only-ft1={sorted(k1 - kb)[:5]}",
    )
    require(
        kb == k2,
        f"base/ft2 tensor names differ: only-base={sorted(kb - k2)[:5]} only-ft2={sorted(k2 - kb)[:5]}",
    )
    require(
        len(kb) == N_TOTAL_EXPECTED,
        f"expected {N_TOTAL_EXPECTED} tensors in each checkpoint, found {len(kb)}",
    )

    mlp_names = sorted(n for n in kb if MLP_RE.match(n))
    require(
        len(mlp_names) == N_MLP_EXPECTED,
        f"expected {N_MLP_EXPECTED} MLP tensors, matched {len(mlp_names)}: {mlp_names[:5]}",
    )
    shared_names = sorted(kb - set(mlp_names))

    for name in sorted(kb):
        b, t1, t2 = base.get(name), ft1.get(name), ft2.get(name)
        for tag, t in (("ft1", t1), ("ft2", t2)):
            require(
                t.shape == b.shape,
                f"{name}: shape mismatch base {tuple(b.shape)} vs {tag} {tuple(t.shape)}",
            )
            require(
                t.dtype == b.dtype,
                f"{name}: dtype mismatch base {b.dtype} vs {tag} {t.dtype}",
            )
        if name in shared_names:
            require(
                torch.equal(b, t1),
                f"non-MLP tensor {name} differs between base and ft1 (frozen backbone violated)",
            )
            require(
                torch.equal(b, t2),
                f"non-MLP tensor {name} differs between base and ft2 (frozen backbone violated)",
            )

    print(f"verified: {len(kb)} tensor names match; {len(shared_names)} non-MLP tensors identical "
          f"in all three checkpoints; {len(mlp_names)} MLP tensors to merge")

    # --- Step 2/3: merge ----------------------------------------------------
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for name in sorted(kb):
        b = base.get(name)
        if name in shared_names:
            out[name] = b.clone()
            continue
        require(b.dtype == torch.float32, f"{name}: expected float32 base tensor, got {b.dtype}")
        b32 = b.to(torch.float32)
        # each task vector is taken against the unmodified base
        tv1 = ft1.get(name).to(torch.float32) - b32
        tv2 = ft2.get(name).to(torch.float32) - b32
        out[name] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).contiguous()
        require(out[name].dtype == torch.float32, f"{name}: merged tensor is not float32")
        merged += 1

    # --- Required checks ----------------------------------------------------
    require(merged == N_MLP_EXPECTED, f"merged {merged} tensors, expected {N_MLP_EXPECTED}")
    require(len(out) == N_TOTAL_EXPECTED, f"output has {len(out)} tensors, expected {N_TOTAL_EXPECTED}")
    require(set(out) == kb, "output tensor names differ from the base tensor names")

    # --- Step 4: write ------------------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".safetensors.tmp")
    save_file(out, str(tmp))
    tmp.replace(OUT)

    # post-write verification of the on-disk artifact
    with safe_open(str(OUT), framework="pt", device="cpu") as f:
        written = set(f.keys())
    require(
        len(written) == N_TOTAL_EXPECTED,
        f"written file has {len(written)} tensors, expected {N_TOTAL_EXPECTED}",
    )
    require(written == kb, "written tensor names differ from the base tensor names")

    print(f"wrote {OUT} with {len(written)} tensors ({merged} merged, {len(shared_names)} copied)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Check as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
