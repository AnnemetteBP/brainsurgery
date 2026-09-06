"""
T4: Task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the 64 MLP
tensors, computed in float32 and cast back to float16. Everything else is copied
unchanged from base. Verifies shared-tensor identity before touching anything.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
OUT_PATH = HERE / "model.safetensors"

LAMBDA = 0.4
N_LAYERS = 16
MLP_SUFFIXES = [
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
]


def mlp_tensor_names() -> set[str]:
    names = set()
    for i in range(N_LAYERS):
        for suffix in MLP_SUFFIXES:
            names.add(f"gpt_neox.layers.{i}.mlp.{suffix.split('.', 1)[1]}")
    return names


def main() -> None:
    base = load_file(INPUTS / "base" / "model.safetensors")
    ft1 = load_file(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_file(INPUTS / "ft2" / "model.safetensors")

    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())
    if not (base_keys == ft1_keys == ft2_keys):
        missing_in_ft1 = base_keys - ft1_keys
        missing_in_ft2 = base_keys - ft2_keys
        extra_in_ft1 = ft1_keys - base_keys
        extra_in_ft2 = ft2_keys - base_keys
        raise SystemExit(
            "Tensor name mismatch between checkpoints:\n"
            f"  missing in ft1: {sorted(missing_in_ft1)}\n"
            f"  missing in ft2: {sorted(missing_in_ft2)}\n"
            f"  extra in ft1: {sorted(extra_in_ft1)}\n"
            f"  extra in ft2: {sorted(extra_in_ft2)}\n"
        )

    if len(base_keys) != 244:
        raise SystemExit(f"Expected 244 tensors in base checkpoint, found {len(base_keys)}")

    mlp_names = mlp_tensor_names()
    if len(mlp_names) != 64:
        raise SystemExit(f"Expected 64 MLP tensor names, computed {len(mlp_names)}")
    unknown_mlp = mlp_names - base_keys
    if unknown_mlp:
        raise SystemExit(f"MLP tensor names not present in checkpoint: {sorted(unknown_mlp)}")

    non_mlp_names = base_keys - mlp_names

    # Step 1: verify everything outside the 64 MLP tensors is bit-identical across
    # all three checkpoints. Abort loudly if the frozen-backbone assumption fails.
    mismatched = []
    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{name}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched.append(f"{name}: values differ outside the MLP tensor set")
    if mismatched:
        raise SystemExit(
            "Non-MLP tensors differ across checkpoints (frozen-backbone assumption "
            "violated); aborting:\n  " + "\n  ".join(mismatched)
        )

    # Step 2: merge the 64 MLP tensors via task arithmetic, in float32, against the
    # unmodified base for both fine-tunes (never against a partially merged result).
    out = {}
    merged_count = 0
    for name in mlp_names:
        b = base[name]
        f1 = ft1[name]
        f2 = ft2[name]
        b32 = b.to(torch.float32)
        merged32 = (
            b32 + LAMBDA * (f1.to(torch.float32) - b32) + LAMBDA * (f2.to(torch.float32) - b32)
        )
        out[name] = merged32.to(b.dtype)
        merged_count += 1

    if merged_count != 64:
        raise SystemExit(f"Expected to merge exactly 64 tensors, merged {merged_count}")

    # Step 3: everything else comes from base, unchanged.
    for name in non_mlp_names:
        out[name] = base[name]

    if len(out) != 244:
        raise SystemExit(f"Expected output with exactly 244 tensors, got {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"Wrote {len(out)} tensors ({merged_count} merged) to {OUT_PATH}")


if __name__ == "__main__":
    main()
