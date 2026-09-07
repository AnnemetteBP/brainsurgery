#!/usr/bin/env python
"""
T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the 48 MLP tensors
out[X] = base[X]                                                       for everything else

Plain script on top of `safetensors` + `torch` (see F-allowed.md): task
arithmetic here is simple enough that hand-writing it is more auditable and
less error-prone than wiring a mergekit YAML config for two direct (non-HF-repo)
safetensors files, while still doing the required verification passes.

Usage: .venv/bin/python solution.py
Reads inputs/{base,ft1,ft2}/model.safetensors, writes out/T4/model.safetensors.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
NUM_LAYERS = 12

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs"
OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "model.safetensors"


def mlp_tensor_names(num_layers: int) -> set[str]:
    names = set()
    for i in range(num_layers):
        names.add(f"h.{i}.mlp.c_fc.weight")
        names.add(f"h.{i}.mlp.c_fc.bias")
        names.add(f"h.{i}.mlp.c_proj.weight")
        names.add(f"h.{i}.mlp.c_proj.bias")
    return names


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def fail(msg: str) -> None:
    print(f"ABORT: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_all(INPUTS / "base" / "model.safetensors")
    ft1 = load_all(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_all(INPUTS / "ft2" / "model.safetensors")

    mlp_names = mlp_tensor_names(NUM_LAYERS)
    if len(mlp_names) != 48:
        fail(f"expected 48 MLP tensor names, computed {len(mlp_names)}")

    # --- Step 1: shared-tensor verification ---
    names_base, names_ft1, names_ft2 = set(base), set(ft1), set(ft2)
    if not (names_base == names_ft1 == names_ft2):
        only_base = names_base - names_ft1 - names_ft2
        only_ft1 = names_ft1 - names_base
        only_ft2 = names_ft2 - names_base
        fail(
            "tensor name sets differ across checkpoints: "
            f"base-only={sorted(only_base)[:5]} ft1-only={sorted(only_ft1)[:5]} "
            f"ft2-only={sorted(only_ft2)[:5]}"
        )

    if not mlp_names.issubset(names_base):
        missing = mlp_names - names_base
        fail(f"expected MLP tensor names missing from checkpoints: {sorted(missing)}")

    non_mlp_names = names_base - mlp_names
    mismatched = []
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{name}: shape mismatch base={tuple(b.shape)} "
                               f"ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched.append(f"{name}: values differ outside the 48 MLP tensors")
    if mismatched:
        fail(
            f"{len(mismatched)} non-MLP tensor(s) are not identical across base/ft1/ft2:\n"
            + "\n".join(mismatched[:20])
        )

    # --- Step 2 & 3: task-arithmetic merge, MLP tensors only ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.to(base[name].dtype).contiguous()
            merged_count += 1
        else:
            out[name] = base[name].clone().contiguous()

    if merged_count != 48:
        fail(f"expected to merge exactly 48 tensors, merged {merged_count}")

    if len(out) != 160:
        fail(f"expected output to have exactly 160 tensors, got {len(out)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_FILE))

    print(f"OK: wrote {len(out)} tensors ({merged_count} merged) to {OUT_FILE}")


if __name__ == "__main__":
    main()
