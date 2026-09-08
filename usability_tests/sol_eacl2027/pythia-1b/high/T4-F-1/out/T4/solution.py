#!/usr/bin/env python3
"""Merge two Pythia-1B MLP fine-tunes by task-vector arithmetic."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


BASE_PATH = Path("inputs/base/model.safetensors")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUTPUT_PATH = Path("out/T4/model.safetensors")
TEMP_PATH = OUTPUT_PATH.with_suffix(".safetensors.tmp")
LAMBDA = 0.4
EXPECTED_TENSOR_COUNT = 244


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


def verify_inputs(mlp_names: set[str]) -> list[str]:
    """Perform every input precondition check before any output is built."""
    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2,
    ):
        base_names = set(base.keys())
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        if len(base_names) != EXPECTED_TENSOR_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_TENSOR_COUNT} input tensors, found {len(base_names)}"
            )
        missing_mlp = mlp_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")
        if len(mlp_names) != 64:
            raise RuntimeError(f"Expected 64 MLP tensor names, generated {len(mlp_names)}")

        ordered_names = sorted(base_names)
        for name in ordered_names:
            base_tensor = base.get_tensor(name)
            ft1_tensor = ft1.get_tensor(name)
            ft2_tensor = ft2.get_tensor(name)
            if (
                base_tensor.shape != ft1_tensor.shape
                or base_tensor.shape != ft2_tensor.shape
                or base_tensor.dtype != ft1_tensor.dtype
                or base_tensor.dtype != ft2_tensor.dtype
            ):
                raise RuntimeError(f"Shape or dtype mismatch for tensor {name}")
            if name in mlp_names and base_tensor.dtype != torch.float16:
                raise RuntimeError(f"Expected float16 MLP tensor {name}, got {base_tensor.dtype}")
            if name not in mlp_names and (
                not torch.equal(base_tensor, ft1_tensor)
                or not torch.equal(base_tensor, ft2_tensor)
            ):
                raise RuntimeError(f"Frozen-backbone verification failed for {name}")

    return ordered_names


def merge(ordered_names: list[str], mlp_names: set[str]) -> None:
    # This happens only after verify_inputs has completed successfully.
    output = load_file(BASE_PATH, device="cpu")
    merged_count = 0

    with (
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2,
    ):
        for name in ordered_names:
            if name not in mlp_names:
                continue
            base32 = output[name].to(torch.float32)
            merged = base32 + LAMBDA * (ft1.get_tensor(name).float() - base32)
            merged.add_(LAMBDA * (ft2.get_tensor(name).float() - base32))
            output[name] = merged.to(torch.float16)
            merged_count += 1

    if merged_count != 64:
        raise RuntimeError(f"Expected to merge exactly 64 tensors, merged {merged_count}")
    if len(output) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TENSOR_COUNT} output tensors, built {len(output)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TEMP_PATH.exists():
        TEMP_PATH.unlink()
    try:
        save_file(output, TEMP_PATH)
        with safe_open(TEMP_PATH, framework="pt", device="cpu") as saved:
            saved_names = set(saved.keys())
            if len(saved_names) != EXPECTED_TENSOR_COUNT:
                raise RuntimeError(
                    f"Saved file has {len(saved_names)} tensors, expected "
                    f"{EXPECTED_TENSOR_COUNT}"
                )
            if saved_names != set(ordered_names):
                raise RuntimeError("Saved output tensor names differ from the inputs")
        os.replace(TEMP_PATH, OUTPUT_PATH)
    finally:
        if TEMP_PATH.exists():
            TEMP_PATH.unlink()


def main() -> None:
    mlp_names = expected_mlp_names()
    ordered_names = verify_inputs(mlp_names)
    merge(ordered_names, mlp_names)
    print(
        f"Wrote {OUTPUT_PATH} with {EXPECTED_TENSOR_COUNT} tensors; "
        "merged exactly 64 MLP tensors."
    )


if __name__ == "__main__":
    main()
