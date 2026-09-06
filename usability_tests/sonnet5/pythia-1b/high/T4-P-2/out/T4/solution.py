"""
T4: Task-vector merge of two fine-tunes (Pythia-1B).

Merges ft1 and ft2 into base by task arithmetic on the 64 MLP tensors:

    out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

with lambda = 0.4, computed in float32 and cast back to float16. Every other
tensor is copied unchanged from base. Verifies before touching anything that
base/ft1/ft2 have identical tensor names and that every non-MLP tensor is
bit-identical across all three checkpoints.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX_ROOT = HERE.parents[1]  # out/T4/solution.py -> out -> sandbox root
INPUTS = SANDBOX_ROOT / "inputs"
OUT_PATH = HERE / "model.safetensors"

LAMBDA = 0.4
NUM_LAYERS = 16
EXPECTED_NUM_TENSORS = 244
EXPECTED_NUM_MLP = 64

MLP_SUFFIXES = [
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
]


def mlp_tensor_names() -> set[str]:
    names = set()
    for i in range(NUM_LAYERS):
        for suffix in MLP_SUFFIXES:
            names.add(f"gpt_neox.layers.{i}.{suffix}")
    return names


def load_all(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base_path = INPUTS / "base" / "model.safetensors"
    ft1_path = INPUTS / "ft1" / "model.safetensors"
    ft2_path = INPUTS / "ft2" / "model.safetensors"

    base = load_all(base_path)
    ft1 = load_all(ft1_path)
    ft2 = load_all(ft2_path)

    # --- Step 1: verify shared structure before touching anything ---
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        missing_ft1 = base_keys - ft1_keys
        missing_ft2 = base_keys - ft2_keys
        extra_ft1 = ft1_keys - base_keys
        extra_ft2 = ft2_keys - base_keys
        raise AssertionError(
            "Tensor name mismatch between base/ft1/ft2: "
            f"missing_in_ft1={sorted(missing_ft1)} extra_in_ft1={sorted(extra_ft1)} "
            f"missing_in_ft2={sorted(missing_ft2)} extra_in_ft2={sorted(extra_ft2)}"
        )

    mlp_names = mlp_tensor_names()
    unexpected_mlp_names = mlp_names - base_keys
    if unexpected_mlp_names:
        raise AssertionError(
            f"Expected MLP tensor names not found in checkpoint: {sorted(unexpected_mlp_names)}"
        )
    if len(mlp_names) != EXPECTED_NUM_MLP:
        raise AssertionError(f"Expected {EXPECTED_NUM_MLP} MLP tensor names, got {len(mlp_names)}")

    non_mlp_names = base_keys - mlp_names
    mismatched = []
    for name in sorted(non_mlp_names):
        b, x1, x2 = base[name], ft1[name], ft2[name]
        if b.shape != x1.shape or b.shape != x2.shape:
            mismatched.append(f"{name}: shape base={tuple(b.shape)} ft1={tuple(x1.shape)} ft2={tuple(x2.shape)}")
            continue
        if b.dtype != x1.dtype or b.dtype != x2.dtype:
            mismatched.append(f"{name}: dtype base={b.dtype} ft1={x1.dtype} ft2={x2.dtype}")
            continue
        if not torch.equal(b, x1):
            mismatched.append(f"{name}: differs between base and ft1 but is not an MLP tensor")
            continue
        if not torch.equal(b, x2):
            mismatched.append(f"{name}: differs between base and ft2 but is not an MLP tensor")
            continue
    if mismatched:
        raise AssertionError(
            "Non-MLP tensors are not identical across base/ft1/ft2 (frozen-backbone "
            "assumption violated):\n" + "\n".join(mismatched)
        )

    # --- Step 2 & 3: merge MLP tensors, copy everything else unchanged ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base_keys:
        if name in mlp_names:
            b32 = base[name].to(torch.float32)
            x1_32 = ft1[name].to(torch.float32)
            x2_32 = ft2[name].to(torch.float32)
            merged32 = b32 + LAMBDA * (x1_32 - b32) + LAMBDA * (x2_32 - b32)
            out[name] = merged32.to(base[name].dtype).contiguous()
            merged_count += 1
        else:
            out[name] = base[name].clone().contiguous()

    if merged_count != EXPECTED_NUM_MLP:
        raise AssertionError(f"Expected to merge {EXPECTED_NUM_MLP} tensors, merged {merged_count}")

    # --- Step 4: write output, exactly 244 tensors ---
    if len(out) != EXPECTED_NUM_TENSORS:
        raise AssertionError(f"Expected {EXPECTED_NUM_TENSORS} output tensors, got {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))

    print(f"Wrote {OUT_PATH} with {len(out)} tensors ({merged_count} merged).")


if __name__ == "__main__":
    sys.exit(main())
