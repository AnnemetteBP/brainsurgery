"""T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])
for the 48 MLP tensors; every other tensor is copied unchanged from base.
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
OUT_PATH = HERE / "model.safetensors"


def mlp_tensor_names() -> set[str]:
    names = set()
    for i in range(12):
        names.add(f"h.{i}.mlp.c_fc.weight")
        names.add(f"h.{i}.mlp.c_fc.bias")
        names.add(f"h.{i}.mlp.c_proj.weight")
        names.add(f"h.{i}.mlp.c_proj.bias")
    return names


def fail(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base_path = INPUTS / "base" / "model.safetensors"
    ft1_path = INPUTS / "ft1" / "model.safetensors"
    ft2_path = INPUTS / "ft2" / "model.safetensors"

    base = load_file(str(base_path))
    ft1 = load_file(str(ft1_path))
    ft2 = load_file(str(ft2_path))

    # --- Step 1: verify shared structure ---
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if base_keys != ft1_keys or base_keys != ft2_keys:
        only_base = base_keys - ft1_keys - ft2_keys
        only_ft1 = ft1_keys - base_keys
        only_ft2 = ft2_keys - base_keys
        fail(
            "tensor name mismatch between checkpoints: "
            f"only_in_base={sorted(only_base)} only_in_ft1={sorted(only_ft1)} "
            f"only_in_ft2={sorted(only_ft2)}"
        )

    mlp_names = mlp_tensor_names()
    missing_mlp = mlp_names - base_keys
    if missing_mlp:
        fail(f"expected MLP tensor names missing from checkpoints: {sorted(missing_mlp)}")

    non_mlp_names = base_keys - mlp_names

    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            fail(f"shape mismatch on shared tensor {name!r}: base={b.shape} ft1={f1.shape} ft2={f2.shape}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            fail(f"dtype mismatch on shared tensor {name!r}: base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
        if not torch.equal(b, f1):
            fail(f"tensor {name!r} is supposed to be shared (non-MLP) but differs between base and ft1")
        if not torch.equal(b, f2):
            fail(f"tensor {name!r} is supposed to be shared (non-MLP) but differs between base and ft2")

    # --- Step 2 & 3: merge MLP tensors, copy everything else ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base_keys:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            if b.shape != f1.shape or b.shape != f2.shape:
                fail(f"shape mismatch on MLP tensor {name!r}: base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.contiguous()
            merged_count += 1
        else:
            out[name] = base[name].clone().contiguous()

    # --- Required checks ---
    if merged_count != 48:
        fail(f"expected exactly 48 merged MLP tensors, got {merged_count}")
    if len(out) != 160:
        fail(f"expected exactly 160 output tensors, got {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {len(out)} tensors ({merged_count} merged) to {OUT_PATH}")


if __name__ == "__main__":
    main()
