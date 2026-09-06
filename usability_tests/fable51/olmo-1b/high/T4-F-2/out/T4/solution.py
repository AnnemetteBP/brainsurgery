"""T4: task-vector merge of two fine-tunes onto OLMo-1B-0724-hf.

Plain torch + safetensors. Steps, in the order the task requires:
  1. verify identical key sets across base / ft1 / ft2, and that every tensor
     outside the 48 MLP tensors is bit-identical in all three (abort otherwise);
  2. out[X] = base[X] + L*(ft1[X]-base[X]) + L*(ft2[X]-base[X]) in float32,
     both task vectors taken against the *unmodified* base;
  3. everything else copied from the base unchanged;
  4. write out/T4/model.safetensors with exactly 114 tensors.
Every check raises (non-zero exit) if it does not hold.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
N_LAYERS = 16
EXPECTED_TOTAL = 114
EXPECTED_MERGED = 48
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


class Checkpoint:
    """Lazy reader over one or more safetensors files (sharded or single)."""

    def __init__(self, files: list[Path]):
        self.handles = [safe_open(str(f), framework="pt", device="cpu") for f in files]
        self.index: dict[str, int] = {}
        for i, h in enumerate(self.handles):
            for k in h.keys():
                if k in self.index:
                    raise RuntimeError(f"duplicate key {k!r} across shards {files}")
                self.index[k] = i

    def keys(self) -> set[str]:
        return set(self.index)

    def get(self, key: str) -> torch.Tensor:
        return self.handles[self.index[key]].get_tensor(key)


def open_base() -> Checkpoint:
    idx = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx["weight_map"].values()))
    ckpt = Checkpoint([BASE_DIR / s for s in shards])
    if ckpt.keys() != set(idx["weight_map"]):
        raise RuntimeError("base index.json weight_map disagrees with shard contents")
    return ckpt


def main() -> int:
    base, ft1, ft2 = open_base(), Checkpoint([FT1]), Checkpoint([FT2])

    # ---- step 1: verification before anything else ------------------------
    kb, k1, k2 = base.keys(), ft1.keys(), ft2.keys()
    if not (kb == k1 == k2):
        raise RuntimeError(
            "tensor name sets differ:\n"
            f"  base-only: {sorted(kb - k1 - k2)}\n"
            f"  ft1-only:  {sorted(k1 - kb)}\n  ft2-only:  {sorted(k2 - kb)}"
        )
    if len(kb) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} tensors, inputs have {len(kb)}")

    expected_mlp = {
        f"model.layers.{i}.mlp.{p}.weight"
        for i in range(N_LAYERS)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    mlp_keys = {k for k in kb if MLP_RE.match(k)}
    if mlp_keys != expected_mlp:
        raise RuntimeError(
            f"MLP key set unexpected: extra={sorted(mlp_keys - expected_mlp)} "
            f"missing={sorted(expected_mlp - mlp_keys)}"
        )
    shared_keys = sorted(kb - mlp_keys)

    for k in shared_keys:
        b, a1, a2 = base.get(k), ft1.get(k), ft2.get(k)
        for name, t in (("ft1", a1), ("ft2", a2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                raise RuntimeError(
                    f"shared tensor {k!r} shape/dtype mismatch in {name}: "
                    f"{tuple(t.shape)}/{t.dtype} vs base {tuple(b.shape)}/{b.dtype}"
                )
            # bit-exact comparison (torch.equal treats NaN != NaN, so compare bytes)
            if not torch.equal(t.view(torch.int32) if t.dtype == torch.float32 else t,
                               b.view(torch.int32) if b.dtype == torch.float32 else b):
                raise RuntimeError(f"shared tensor {k!r} differs between base and {name}")
    print(f"[verify] {len(shared_keys)} shared tensors bit-identical in base/ft1/ft2")

    # ---- step 2 + 3: merge MLP tensors, copy the rest ----------------------
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for k in sorted(kb):
        b = base.get(k)
        if k in mlp_keys:
            a1, a2 = ft1.get(k), ft2.get(k)
            if not (b.dtype == a1.dtype == a2.dtype == torch.float32):
                raise RuntimeError(f"MLP tensor {k!r} is not float32 in all inputs")
            if not (b.shape == a1.shape == a2.shape):
                raise RuntimeError(f"MLP tensor {k!r} shape mismatch across inputs")
            # both task vectors are against the *unmodified* base tensor `b`
            tv1 = a1 - b
            tv2 = a2 - b
            out[k] = (b + LAMBDA * tv1 + LAMBDA * tv2).contiguous()
            merged += 1
        else:
            out[k] = b.contiguous()

    if merged != EXPECTED_MERGED:
        raise RuntimeError(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        raise RuntimeError(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")
    print(f"[merge] merged {merged} MLP tensors with lambda={LAMBDA}")

    # ---- step 4: write and re-verify from disk -----------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})
    del out

    with safe_open(str(OUT), framework="pt", device="cpu") as f:
        written = set(f.keys())
        if len(written) != EXPECTED_TOTAL or written != kb:
            raise RuntimeError(f"written file has {len(written)} tensors / wrong key set")
        for k in shared_keys:
            if not torch.equal(f.get_tensor(k), base.get(k)):
                raise RuntimeError(f"written shared tensor {k!r} differs from base")
        # spot-check one merged tensor against a fresh recomputation
        k = "model.layers.0.mlp.gate_proj.weight"
        b = base.get(k)
        ref = b + LAMBDA * (ft1.get(k) - b) + LAMBDA * (ft2.get(k) - b)
        got = f.get_tensor(k)
        rel = (got - ref).norm() / ref.norm()
        if not torch.isfinite(rel) or rel > 1e-6:
            raise RuntimeError(f"merged tensor {k!r} rel. Frobenius error {rel:.3e} too large")
    print(f"[done] wrote {OUT} with {EXPECTED_TOTAL} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
