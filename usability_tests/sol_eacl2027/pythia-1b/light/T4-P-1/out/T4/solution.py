from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUTPUT = Path("out/T4/model.safetensors")
LAMBDA = 0.4


def mlp_names() -> set[str]:
    suffixes = (
        "dense_h_to_4h.weight",
        "dense_h_to_4h.bias",
        "dense_4h_to_h.weight",
        "dense_4h_to_h.bias",
    )
    return {
        f"gpt_neox.layers.{layer}.mlp.{suffix}"
        for layer in range(16)
        for suffix in suffixes
    }


def main() -> None:
    merge_names = mlp_names()
    if len(merge_names) != 64:
        raise RuntimeError(f"Expected 64 MLP tensor names, constructed {len(merge_names)}")

    with safe_open(BASE, framework="pt", device="cpu") as base, \
         safe_open(FT1, framework="pt", device="cpu") as ft1, \
         safe_open(FT2, framework="pt", device="cpu") as ft2:
        base_names = set(base.keys())
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())

        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError("The three checkpoints do not have identical tensor names")
        missing = merge_names - base_names
        if missing:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing)}")

        # Complete the frozen-backbone verification before constructing output.
        for name in sorted(base_names - merge_names):
            base_tensor = base.get_tensor(name)
            if not torch.equal(base_tensor, ft1.get_tensor(name)):
                raise RuntimeError(f"Fine-tune 1 changed non-MLP tensor: {name}")
            if not torch.equal(base_tensor, ft2.get_tensor(name)):
                raise RuntimeError(f"Fine-tune 2 changed non-MLP tensor: {name}")

        output = {}
        merged = 0
        for name in sorted(base_names):
            base_tensor = base.get_tensor(name)
            if name in merge_names:
                ft1_tensor = ft1.get_tensor(name)
                ft2_tensor = ft2.get_tensor(name)
                if (ft1_tensor.shape != base_tensor.shape or
                        ft2_tensor.shape != base_tensor.shape or
                        ft1_tensor.dtype != base_tensor.dtype or
                        ft2_tensor.dtype != base_tensor.dtype):
                    raise RuntimeError(f"Shape or dtype mismatch for {name}")
                base32 = base_tensor.float()
                output[name] = (
                    base32
                    + LAMBDA * (ft1_tensor.float() - base32)
                    + LAMBDA * (ft2_tensor.float() - base32)
                ).to(base_tensor.dtype)
                merged += 1
            else:
                output[name] = base_tensor

    if merged != 64:
        raise RuntimeError(f"Expected to merge 64 tensors, merged {merged}")
    if len(output) != 244:
        raise RuntimeError(f"Expected 244 output tensors, got {len(output)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT)
    print(f"Saved {len(output)} tensors to {OUTPUT} ({merged} merged)")


if __name__ == "__main__":
    main()
