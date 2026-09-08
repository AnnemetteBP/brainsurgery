#!/usr/bin/env python3
"""Merge two Pythia fine-tunes by task-vector arithmetic."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUTPUT = Path("out/T4/model.safetensors")
LAMBDA = 0.4


def expected_mlp_names() -> set[str]:
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
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 64:
        raise RuntimeError(f"Expected 64 generated MLP names, got {len(mlp_names)}")

    # Complete all precondition checks before constructing any output tensors.
    with safe_open(BASE, framework="pt", device="cpu") as base_file, safe_open(
        FT1, framework="pt", device="cpu"
    ) as ft1_file, safe_open(FT2, framework="pt", device="cpu") as ft2_file:
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())
        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "Checkpoint tensor-name sets differ: "
                f"base={len(base_names)}, ft1={len(ft1_names)}, ft2={len(ft2_names)}"
            )
        missing_mlp = mlp_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")

        for name in sorted(base_names - mlp_names):
            base_tensor = base_file.get_tensor(name)
            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            if not torch.equal(base_tensor, ft1_tensor):
                raise RuntimeError(f"Non-MLP tensor differs between base and ft1: {name}")
            if not torch.equal(base_tensor, ft2_tensor):
                raise RuntimeError(f"Non-MLP tensor differs between base and ft2: {name}")

    output_tensors: dict[str, torch.Tensor] = {}
    merged_count = 0
    with safe_open(BASE, framework="pt", device="cpu") as base_file, safe_open(
        FT1, framework="pt", device="cpu"
    ) as ft1_file, safe_open(FT2, framework="pt", device="cpu") as ft2_file:
        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            if name not in mlp_names:
                output_tensors[name] = base_tensor
                continue

            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            if (
                ft1_tensor.shape != base_tensor.shape
                or ft2_tensor.shape != base_tensor.shape
                or ft1_tensor.dtype != base_tensor.dtype
                or ft2_tensor.dtype != base_tensor.dtype
            ):
                raise RuntimeError(f"Shape or dtype mismatch for MLP tensor: {name}")

            base_f32 = base_tensor.float()
            merged = (
                base_f32
                + LAMBDA * (ft1_tensor.float() - base_f32)
                + LAMBDA * (ft2_tensor.float() - base_f32)
            )
            output_tensors[name] = merged.to(base_tensor.dtype)
            merged_count += 1

    if merged_count != 64:
        raise RuntimeError(f"Expected to merge 64 tensors, merged {merged_count}")
    if len(output_tensors) != 244:
        raise RuntimeError(f"Expected 244 output tensors, got {len(output_tensors)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, OUTPUT)

    # Validate the serialized artifact, not only the in-memory mapping.
    with safe_open(OUTPUT, framework="pt", device="cpu") as output_file:
        written_names = set(output_file.keys())
    if written_names != base_names or len(written_names) != 244:
        raise RuntimeError(
            f"Serialized output has wrong key set/count: {len(written_names)} tensors"
        )

    print(f"Wrote {OUTPUT} with {len(written_names)} tensors; merged {merged_count} MLP tensors")


if __name__ == "__main__":
    main()
