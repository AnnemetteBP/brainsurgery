from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


LAMBDA = 0.4
ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
FT1_PATH = ROOT / "inputs/ft1/model.safetensors"
FT2_PATH = ROOT / "inputs/ft2/model.safetensors"
OUTPUT_PATH = Path(__file__).resolve().parent / "model.safetensors"


def mlp_names() -> set[str]:
    suffixes = (
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    )
    return {f"h.{layer}.{suffix}" for layer in range(12) for suffix in suffixes}


def main() -> None:
    merge_names = mlp_names()
    if len(merge_names) != 48:
        raise RuntimeError(f"Expected 48 MLP tensor names, got {len(merge_names)}")

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base_file,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1_file,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2_file,
    ):
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())

        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "Checkpoint tensor-name sets differ: "
                f"base={len(base_names)}, ft1={len(ft1_names)}, ft2={len(ft2_names)}"
            )
        missing_mlp = merge_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")

        # Validate all layouts, and validate the frozen backbone bit-for-bit,
        # before computing any task-vector tensors.
        for name in sorted(base_names):
            base = base_file.get_tensor(name)
            ft1 = ft1_file.get_tensor(name)
            ft2 = ft2_file.get_tensor(name)
            if base.shape != ft1.shape or base.shape != ft2.shape:
                raise RuntimeError(f"Shape mismatch for {name}")
            if base.dtype != ft1.dtype or base.dtype != ft2.dtype:
                raise RuntimeError(f"Dtype mismatch for {name}")
            if name not in merge_names and (
                not torch.equal(base, ft1) or not torch.equal(base, ft2)
            ):
                raise RuntimeError(f"Frozen tensor differs from base: {name}")

        merged: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base = base_file.get_tensor(name)
            if name in merge_names:
                ft1 = ft1_file.get_tensor(name)
                ft2 = ft2_file.get_tensor(name)
                if base.dtype != torch.float32:
                    raise RuntimeError(f"MLP tensor is not float32: {name} ({base.dtype})")
                merged[name] = (
                    base
                    + LAMBDA * (ft1 - base)
                    + LAMBDA * (ft2 - base)
                )
                merged_count += 1
            else:
                merged[name] = base

        if merged_count != 48:
            raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
        if len(merged) != 160:
            raise RuntimeError(f"Expected 160 output tensors, got {len(merged)}")

        save_file(merged, OUTPUT_PATH)

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as output_file:
        output_names = set(output_file.keys())
        if len(output_names) != 160:
            raise RuntimeError(
                f"Saved output should contain 160 tensors, got {len(output_names)}"
            )
        if output_names != base_names:
            raise RuntimeError("Saved output tensor names differ from the base")

    print(f"Wrote {len(output_names)} tensors to {OUTPUT_PATH}")
    print(f"Merged {merged_count} MLP tensors with lambda={LAMBDA}")


if __name__ == "__main__":
    main()
