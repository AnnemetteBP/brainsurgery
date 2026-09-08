#!/usr/bin/env python3
"""Merge two Pythia task vectors into their shared base checkpoint."""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUTPUT = Path("out/T4/model.safetensors")
SCALE = 0.4
EXPECTED_TENSORS = 244


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


def bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare tensor storage values exactly, including NaN and signed-zero bits."""
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def verify_inputs(mlp_names: set[str]) -> tuple[list[str], dict[str, str] | None]:
    """Verify every precondition before any output tensor is constructed or written."""
    with (
        safe_open(BASE, framework="pt", device="cpu") as base,
        safe_open(FT1, framework="pt", device="cpu") as ft1,
        safe_open(FT2, framework="pt", device="cpu") as ft2,
    ):
        base_names = set(base.keys())
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "checkpoint tensor names differ: "
                f"base={len(base_names)}, ft1={len(ft1_names)}, ft2={len(ft2_names)}"
            )
        if len(base_names) != EXPECTED_TENSORS:
            raise RuntimeError(
                f"base has {len(base_names)} tensors, expected {EXPECTED_TENSORS}"
            )
        missing_mlp = mlp_names - base_names
        if missing_mlp or len(mlp_names) != 64:
            raise RuntimeError(
                f"invalid MLP target set: count={len(mlp_names)}, missing={sorted(missing_mlp)}"
            )

        ordered_names = sorted(base_names)
        for name in ordered_names:
            signatures = []
            for checkpoint in (base, ft1, ft2):
                tensor_slice = checkpoint.get_slice(name)
                signatures.append(
                    (tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
                )
            if not (signatures[0] == signatures[1] == signatures[2]):
                raise RuntimeError(f"shape/dtype mismatch for {name}: {signatures}")

            if name not in mlp_names:
                base_tensor = base.get_tensor(name)
                ft1_tensor = ft1.get_tensor(name)
                ft2_tensor = ft2.get_tensor(name)
                if not bits_equal(base_tensor, ft1_tensor):
                    raise RuntimeError(f"ft1 differs from base outside MLP: {name}")
                if not bits_equal(base_tensor, ft2_tensor):
                    raise RuntimeError(f"ft2 differs from base outside MLP: {name}")
                del base_tensor, ft1_tensor, ft2_tensor

        return ordered_names, base.metadata()


def merge(ordered_names: list[str], mlp_names: set[str]) -> dict[str, torch.Tensor]:
    merged_state: dict[str, torch.Tensor] = {}
    merged_count = 0

    with (
        safe_open(BASE, framework="pt", device="cpu") as base,
        safe_open(FT1, framework="pt", device="cpu") as ft1,
        safe_open(FT2, framework="pt", device="cpu") as ft2,
    ):
        for name in ordered_names:
            base_tensor = base.get_tensor(name)
            if name not in mlp_names:
                merged_state[name] = base_tensor
                continue

            # Keep base_f32 unchanged so both task vectors use the original base.
            base_f32 = base_tensor.float()
            result = base_f32 + SCALE * (ft1.get_tensor(name).float() - base_f32)
            result.add_(ft2.get_tensor(name).float() - base_f32, alpha=SCALE)
            merged_state[name] = result.to(dtype=base_tensor.dtype)
            merged_count += 1
            del base_tensor, base_f32, result

    if merged_count != 64:
        raise RuntimeError(f"merged {merged_count} tensors, expected exactly 64")
    if len(merged_state) != EXPECTED_TENSORS:
        raise RuntimeError(
            f"constructed {len(merged_state)} output tensors, expected {EXPECTED_TENSORS}"
        )
    return merged_state


def verify_saved_output(mlp_names: set[str]) -> None:
    """Ensure serialization preserved the exact key set and untouched tensors."""
    with (
        safe_open(BASE, framework="pt", device="cpu") as base,
        safe_open(OUTPUT, framework="pt", device="cpu") as output,
    ):
        base_names = set(base.keys())
        output_names = set(output.keys())
        if output_names != base_names:
            raise RuntimeError("saved output tensor names differ from the base")
        if len(output_names) != EXPECTED_TENSORS:
            raise RuntimeError(
                f"saved output has {len(output_names)} tensors, expected {EXPECTED_TENSORS}"
            )
        if len(output_names & mlp_names) != 64:
            raise RuntimeError("saved output does not contain exactly 64 merged tensors")

        for name in sorted(output_names - mlp_names):
            if not bits_equal(base.get_tensor(name), output.get_tensor(name)):
                raise RuntimeError(f"saved non-MLP tensor changed: {name}")


def main() -> None:
    mlp_names = expected_mlp_names()
    ordered_names, metadata = verify_inputs(mlp_names)
    print("Input verification passed: identical names/layout and 180 shared tensors")

    merged_state = merge(ordered_names, mlp_names)
    temporary = OUTPUT.with_suffix(".safetensors.tmp")
    save_file(merged_state, temporary, metadata=metadata)
    os.replace(temporary, OUTPUT)
    del merged_state
    gc.collect()

    verify_saved_output(mlp_names)
    print(f"Wrote and verified {OUTPUT}: {EXPECTED_TENSORS} tensors, 64 merged")


if __name__ == "__main__":
    main()
