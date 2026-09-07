"""T4: task-vector merge of two Pythia-1B fine-tunes into the base."""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(HERE, "model.safetensors")

LAMBDA = 0.4
N_LAYERS = 16
MLP_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$"
)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(BASE)
    ft1 = load_file(FT1)
    ft2 = load_file(FT2)

    # --- Step 1: verification before touching anything ---
    names = set(base)
    if set(ft1) != names:
        fail(f"ft1 key set differs from base: {set(ft1) ^ names}")
    if set(ft2) != names:
        fail(f"ft2 key set differs from base: {set(ft2) ^ names}")
    if len(names) != 244:
        fail(f"expected 244 tensors in base, found {len(names)}")

    mlp_names = {n for n in names if MLP_RE.match(n)}
    expected_mlp = {
        f"gpt_neox.layers.{i}.mlp.dense_{d}.{p}"
        for i in range(N_LAYERS)
        for d in ("h_to_4h", "4h_to_h")
        for p in ("weight", "bias")
    }
    if mlp_names != expected_mlp:
        fail(f"MLP tensor set mismatch: {mlp_names ^ expected_mlp}")
    if len(mlp_names) != 64:
        fail(f"expected 64 MLP tensors, found {len(mlp_names)}")

    for n in sorted(names):
        for tag, ft in (("ft1", ft1), ("ft2", ft2)):
            if ft[n].shape != base[n].shape:
                fail(f"{tag}[{n}] shape {tuple(ft[n].shape)} != base {tuple(base[n].shape)}")
            if ft[n].dtype != base[n].dtype:
                fail(f"{tag}[{n}] dtype {ft[n].dtype} != base {base[n].dtype}")
        if n in mlp_names:
            continue
        if not torch.equal(base[n], ft1[n]):
            fail(f"non-MLP tensor differs between base and ft1: {n}")
        if not torch.equal(base[n], ft2[n]):
            fail(f"non-MLP tensor differs between base and ft2: {n}")
    print(f"verified: {len(names)} shared names, {len(names) - len(mlp_names)} "
          f"non-MLP tensors identical across base/ft1/ft2")

    # --- Step 2/3: merge; every task vector taken against the unmodified base ---
    out = {}
    merged = 0
    for n in names:
        b = base[n]
        if n in mlp_names:
            b32 = b.to(torch.float32)
            tv1 = ft1[n].to(torch.float32) - b32
            tv2 = ft2[n].to(torch.float32) - b32
            out[n] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).to(b.dtype).contiguous()
            merged += 1
        else:
            out[n] = b.contiguous()

    if merged != 64:
        fail(f"expected to merge 64 tensors, merged {merged}")
    if len(out) != 244:
        fail(f"output has {len(out)} tensors, expected 244")

    save_file(out, OUT, metadata={"format": "pt"})

    # --- Post-save check on the file actually written ---
    reloaded = load_file(OUT)
    if len(reloaded) != 244:
        fail(f"written file has {len(reloaded)} tensors, expected 244")
    if set(reloaded) != names:
        fail("written file key set differs from base")
    for n in names:
        if reloaded[n].dtype != base[n].dtype or reloaded[n].shape != base[n].shape:
            fail(f"written tensor {n} has wrong shape/dtype")
    print(f"OK: wrote {OUT} with {len(reloaded)} tensors, {merged} merged (lambda={LAMBDA})")


if __name__ == "__main__":
    main()
