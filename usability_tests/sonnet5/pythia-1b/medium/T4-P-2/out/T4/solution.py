"""
T4: Task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])
for the 64 MLP tensors, computed in float32 and cast back to float16.
Everything else is copied unchanged from base.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
NUM_LAYERS = 16

BASE_PATH = Path("inputs/base/model.safetensors")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUT_PATH = Path("out/T4/model.safetensors")


def mlp_tensor_names():
    names = set()
    for i in range(NUM_LAYERS):
        prefix = f"gpt_neox.layers.{i}.mlp"
        names.add(f"{prefix}.dense_h_to_4h.weight")
        names.add(f"{prefix}.dense_h_to_4h.bias")
        names.add(f"{prefix}.dense_4h_to_h.weight")
        names.add(f"{prefix}.dense_4h_to_h.bias")
    return names


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    mlp_names = mlp_tensor_names()
    assert len(mlp_names) == 64, f"expected 64 MLP tensor names, got {len(mlp_names)}"

    base = load_all(BASE_PATH)
    ft1 = load_all(FT1_PATH)
    ft2 = load_all(FT2_PATH)

    # Step 1: same tensor names across all three checkpoints.
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())
    if not (base_keys == ft1_keys == ft2_keys):
        missing_in_ft1 = base_keys - ft1_keys
        missing_in_ft2 = base_keys - ft2_keys
        extra_in_ft1 = ft1_keys - base_keys
        extra_in_ft2 = ft2_keys - base_keys
        sys.exit(
            "Tensor name sets differ between checkpoints.\n"
            f"missing_in_ft1={missing_in_ft1}\nmissing_in_ft2={missing_in_ft2}\n"
            f"extra_in_ft1={extra_in_ft1}\nextra_in_ft2={extra_in_ft2}"
        )

    if not mlp_names.issubset(base_keys):
        sys.exit(f"Expected MLP tensor names not found in checkpoint: {mlp_names - base_keys}")

    non_mlp_names = base_keys - mlp_names

    # Verify every non-MLP tensor is bit-identical across all three checkpoints.
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            sys.exit(f"Shape mismatch outside MLP tensors for '{name}': "
                      f"base={b.shape} ft1={f1.shape} ft2={f2.shape}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            sys.exit(f"Dtype mismatch outside MLP tensors for '{name}': "
                      f"base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
        if not torch.equal(b, f1):
            sys.exit(f"Non-MLP tensor '{name}' differs between base and ft1; aborting.")
        if not torch.equal(b, f2):
            sys.exit(f"Non-MLP tensor '{name}' differs between base and ft2; aborting.")

    # Step 2: merge the 64 MLP tensors via task arithmetic, against the
    # unmodified base (not against each other or a running merge).
    output: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            sys.exit(f"Shape mismatch for MLP tensor '{name}': "
                      f"base={b.shape} ft1={f1.shape} ft2={f2.shape}")
        dtype = b.dtype
        b32 = b.float()
        f1_32 = f1.float()
        f2_32 = f2.float()
        merged = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
        output[name] = merged.to(dtype).contiguous()
        merged_count += 1

    assert merged_count == 64, f"expected to merge 64 tensors, merged {merged_count}"

    # Step 3: everything else comes straight from base, unchanged.
    for name in non_mlp_names:
        output[name] = base[name].contiguous()

    if len(output) != 244:
        sys.exit(f"Expected 244 output tensors, got {len(output)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(OUT_PATH))
    print(f"Wrote {len(output)} tensors ({merged_count} merged) to {OUT_PATH}")


if __name__ == "__main__":
    main()
