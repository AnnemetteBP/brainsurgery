from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUTPUT = Path("out/T4/model.safetensors")
LAMBDA = 0.4


def mlp_tensor_names() -> set[str]:
    suffixes = (
        "c_fc.weight",
        "c_fc.bias",
        "c_proj.weight",
        "c_proj.bias",
    )
    return {f"h.{layer}.mlp.{suffix}" for layer in range(12) for suffix in suffixes}


def main() -> None:
    merged_names = mlp_tensor_names()
    if len(merged_names) != 48:
        raise RuntimeError(f"Expected 48 MLP tensor names, got {len(merged_names)}")

    # Complete all precondition checks before computing any merged tensor.
    with safe_open(BASE, framework="pt", device="cpu") as base_file, safe_open(
        FT1, framework="pt", device="cpu"
    ) as ft1_file, safe_open(FT2, framework="pt", device="cpu") as ft2_file:
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())
        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError("The three checkpoints do not have identical tensor names")
        if not merged_names <= base_names:
            missing = sorted(merged_names - base_names)
            raise RuntimeError(f"Missing expected MLP tensors: {missing}")

        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            if base_tensor.shape != ft1_tensor.shape or base_tensor.shape != ft2_tensor.shape:
                raise RuntimeError(f"Shape mismatch for tensor {name}")
            if base_tensor.dtype != ft1_tensor.dtype or base_tensor.dtype != ft2_tensor.dtype:
                raise RuntimeError(f"Dtype mismatch for tensor {name}")
            if name not in merged_names and (
                not torch.equal(base_tensor, ft1_tensor)
                or not torch.equal(base_tensor, ft2_tensor)
            ):
                raise RuntimeError(f"Non-MLP tensor differs from base: {name}")

    output_tensors: dict[str, torch.Tensor] = {}
    merged_count = 0
    with safe_open(BASE, framework="pt", device="cpu") as base_file, safe_open(
        FT1, framework="pt", device="cpu"
    ) as ft1_file, safe_open(FT2, framework="pt", device="cpu") as ft2_file:
        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            if name in merged_names:
                if base_tensor.dtype != torch.float32:
                    raise RuntimeError(f"MLP tensor is not float32: {name}")
                ft1_tensor = ft1_file.get_tensor(name)
                ft2_tensor = ft2_file.get_tensor(name)
                output_tensors[name] = (
                    base_tensor
                    + LAMBDA * (ft1_tensor - base_tensor)
                    + LAMBDA * (ft2_tensor - base_tensor)
                )
                merged_count += 1
            else:
                output_tensors[name] = base_tensor

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
    if len(output_tensors) != 160:
        raise RuntimeError(f"Expected 160 output tensors, got {len(output_tensors)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, OUTPUT)
    print(f"Saved {len(output_tensors)} tensors to {OUTPUT} ({merged_count} merged)")


if __name__ == "__main__":
    main()
