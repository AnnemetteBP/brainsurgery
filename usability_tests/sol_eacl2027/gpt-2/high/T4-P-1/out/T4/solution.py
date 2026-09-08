from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUTPUT_PATH = Path("out/T4/model.safetensors")
LAMBDA = 0.4

MLP_SUFFIXES = (
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
)
MLP_NAMES = {
    f"h.{layer}.{suffix}"
    for layer in range(12)
    for suffix in MLP_SUFFIXES
}


def main() -> None:
    if len(MLP_NAMES) != 48:
        raise RuntimeError(f"Expected 48 MLP tensor names, got {len(MLP_NAMES)}")

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base_file,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1_file,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2_file,
    ):
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())

        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError("The three checkpoints do not have identical tensor names")
        if not MLP_NAMES.issubset(base_names):
            missing = sorted(MLP_NAMES - base_names)
            raise RuntimeError(f"Missing expected MLP tensors: {missing}")

        # Validate the complete shared layout and frozen-backbone assumption
        # before constructing or writing any output tensors.
        for name in sorted(base_names):
            base = base_file.get_tensor(name)
            ft1 = ft1_file.get_tensor(name)
            ft2 = ft2_file.get_tensor(name)
            if base.shape != ft1.shape or base.shape != ft2.shape:
                raise RuntimeError(f"Shape mismatch for tensor {name}")
            if base.dtype != ft1.dtype or base.dtype != ft2.dtype:
                raise RuntimeError(f"Dtype mismatch for tensor {name}")
            if name not in MLP_NAMES and (
                not torch.equal(base, ft1) or not torch.equal(base, ft2)
            ):
                raise RuntimeError(f"Non-MLP tensor differs from base: {name}")

        output = {}
        merged_count = 0
        for name in sorted(base_names):
            base = base_file.get_tensor(name)
            if name in MLP_NAMES:
                ft1 = ft1_file.get_tensor(name)
                ft2 = ft2_file.get_tensor(name)
                if base.dtype != torch.float32:
                    raise RuntimeError(f"MLP tensor is not float32: {name}")
                output[name] = (
                    base
                    + LAMBDA * (ft1 - base)
                    + LAMBDA * (ft2 - base)
                )
                merged_count += 1
            else:
                output[name] = base

        if merged_count != 48:
            raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
        if len(output) != 160:
            raise RuntimeError(f"Expected 160 output tensors, got {len(output)}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, OUTPUT_PATH)

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved:
        if len(saved.keys()) != 160:
            raise RuntimeError(
                f"Saved output has {len(saved.keys())} tensors instead of 160"
            )


if __name__ == "__main__":
    main()
