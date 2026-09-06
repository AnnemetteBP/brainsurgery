"""T4: task-vector merge of two GPT-2 fine-tunes into the base (lambda = 0.4)."""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def main() -> None:
    base = load_file(str(BASE))
    ft1 = load_file(str(FT1))
    ft2 = load_file(str(FT2))

    # Step 1: same names in all three checkpoints.
    names = set(base)
    if names != set(ft1) or names != set(ft2):
        fail(
            "tensor name sets differ: "
            f"base^ft1={sorted(names ^ set(ft1))[:5]} base^ft2={sorted(names ^ set(ft2))[:5]}"
        )
    if len(names) != 160:
        fail(f"expected 160 tensors in base, found {len(names)}")

    mlp_names = sorted(n for n in names if MLP_RE.match(n))
    if len(mlp_names) != 48:
        fail(f"expected 48 MLP tensors, matched {len(mlp_names)}")

    # Step 1: every non-MLP tensor is bit-identical across the three checkpoints.
    for n in sorted(names):
        for tag, other in (("ft1", ft1), ("ft2", ft2)):
            if other[n].shape != base[n].shape or other[n].dtype != base[n].dtype:
                fail(f"{n}: shape/dtype mismatch between base and {tag}")
        if n in mlp_names:
            continue
        for tag, other in (("ft1", ft1), ("ft2", ft2)):
            if not torch.equal(base[n], other[n]):
                fail(f"shared tensor {n} differs between base and {tag}")

    # Step 2: merge, each task vector taken against the untouched base.
    out = {}
    merged = 0
    for n in sorted(names):
        b = base[n]
        if n in mlp_names:
            if b.dtype != torch.float32:
                fail(f"{n}: expected float32, found {b.dtype}")
            tv1 = ft1[n].float() - b
            tv2 = ft2[n].float() - b
            out[n] = (b + LAMBDA * tv1 + LAMBDA * tv2).to(torch.float32).contiguous()
            merged += 1
        else:
            out[n] = b.contiguous()

    if merged != 48:
        fail(f"merged {merged} tensors, expected 48")
    if len(out) != 160:
        fail(f"output has {len(out)} tensors, expected 160")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # Re-read and verify the written file.
    check = load_file(str(OUT))
    if len(check) != 160:
        fail(f"written file has {len(check)} tensors, expected 160")
    if set(check) != names:
        fail("written file key set differs from base")
    for n in names:
        if n not in mlp_names and not torch.equal(check[n], base[n]):
            fail(f"written unchanged tensor {n} differs from base")
    print(f"OK: wrote {OUT} with {len(check)} tensors, {merged} merged (lambda={LAMBDA})")


if __name__ == "__main__":
    main()
