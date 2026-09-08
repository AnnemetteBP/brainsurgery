from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUTPUT_PATH = Path("out/T4/model.safetensors")
LAMBDA = 0.4


def mlp_tensor_names() -> set[str]:
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
    merge_names = mlp_tensor_names()
    assert len(merge_names) == 64, "Internal error: expected 64 MLP tensor names"

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base_file,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1_file,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2_file,
    ):
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())

        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError("Checkpoint tensor names do not match")
        if not merge_names.issubset(base_names):
            missing = sorted(merge_names - base_names)
            raise RuntimeError(f"Missing expected MLP tensors: {missing}")

        # Complete the frozen-backbone verification before constructing output.
        for name in sorted(base_names - merge_names):
            base = base_file.get_tensor(name)
            ft1 = ft1_file.get_tensor(name)
            ft2 = ft2_file.get_tensor(name)
            if not torch.equal(base, ft1) or not torch.equal(base, ft2):
                raise RuntimeError(
                    f"Frozen-backbone check failed: non-MLP tensor differs: {name}"
                )

        output = {}
        merged_count = 0
        for name in sorted(base_names):
            base = base_file.get_tensor(name)
            if name in merge_names:
                ft1 = ft1_file.get_tensor(name)
                ft2 = ft2_file.get_tensor(name)
                if base.shape != ft1.shape or base.shape != ft2.shape:
                    raise RuntimeError(f"Shape mismatch for MLP tensor: {name}")
                merged = (
                    base.float()
                    + LAMBDA * (ft1.float() - base.float())
                    + LAMBDA * (ft2.float() - base.float())
                )
                output[name] = merged.to(dtype=base.dtype)
                merged_count += 1
            else:
                output[name] = base

    if merged_count != 64:
        raise RuntimeError(f"Expected to merge 64 tensors, merged {merged_count}")
    if len(output) != 244:
        raise RuntimeError(f"Expected 244 output tensors, got {len(output)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT_PATH)
    print(f"Wrote {len(output)} tensors to {OUTPUT_PATH} ({merged_count} merged)")


if __name__ == "__main__":
    main()
