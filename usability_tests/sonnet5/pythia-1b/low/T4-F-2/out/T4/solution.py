"""T4: Task-vector merge of two Pythia-1B fine-tunes.

Plain script on top of `safetensors` + `torch` (see REPORT.md for why).
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
OUT_PATH = HERE / "model.safetensors"

LAMBDA = 0.4
NUM_LAYERS = 16
MLP_SUFFIXES = [
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
]


def expected_mlp_keys() -> set[str]:
    keys = set()
    for i in range(NUM_LAYERS):
        for suffix in MLP_SUFFIXES:
            keys.add(f"gpt_neox.layers.{i}.{suffix}")
    return keys


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

    # --- Step 1: verify shared structure and non-MLP tensor identity ---
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if not (base_keys == ft1_keys == ft2_keys):
        missing_ft1 = base_keys ^ ft1_keys
        missing_ft2 = base_keys ^ ft2_keys
        raise SystemExit(
            "Tensor name sets differ between checkpoints.\n"
            f"base ^ ft1: {sorted(missing_ft1)}\n"
            f"base ^ ft2: {sorted(missing_ft2)}"
        )

    mlp_keys = expected_mlp_keys()
    if not mlp_keys.issubset(base_keys):
        missing = mlp_keys - base_keys
        raise SystemExit(f"Expected MLP tensors missing from base checkpoint: {sorted(missing)}")

    non_mlp_keys = base_keys - mlp_keys

    mismatches = []
    for key in non_mlp_keys:
        b = base[key]
        f1 = ft1[key]
        f2 = ft2[key]
        if b.shape != f1.shape or b.dtype != f1.dtype or not torch.equal(b, f1):
            mismatches.append(f"{key} (base vs ft1)")
        if b.shape != f2.shape or b.dtype != f2.dtype or not torch.equal(b, f2):
            mismatches.append(f"{key} (base vs ft2)")
    if mismatches:
        raise SystemExit(
            "Non-MLP tensors differ between checkpoints (frozen-backbone "
            "assumption violated):\n" + "\n".join(mismatches)
        )

    # Also verify MLP tensor shapes/dtypes match across checkpoints.
    for key in mlp_keys:
        b, f1, f2 = base[key], ft1[key], ft2[key]
        if not (b.shape == f1.shape == f2.shape):
            raise SystemExit(f"Shape mismatch for MLP tensor {key}")
        if not (b.dtype == f1.dtype == f2.dtype):
            raise SystemExit(f"Dtype mismatch for MLP tensor {key}")

    # --- Step 2: task-arithmetic merge for the 64 MLP tensors ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for key in base_keys:
        if key in mlp_keys:
            base_dtype = base[key].dtype
            b32 = base[key].to(torch.float32)
            f1_32 = ft1[key].to(torch.float32)
            f2_32 = ft2[key].to(torch.float32)
            merged = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
            out[key] = merged.to(base_dtype)
            merged_count += 1
        else:
            out[key] = base[key].clone()

    if merged_count != 64:
        raise SystemExit(f"Expected to merge exactly 64 tensors, merged {merged_count}")

    if len(out) != 244:
        raise SystemExit(f"Expected output to have exactly 244 tensors, got {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"Wrote {len(out)} tensors ({merged_count} merged) to {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
