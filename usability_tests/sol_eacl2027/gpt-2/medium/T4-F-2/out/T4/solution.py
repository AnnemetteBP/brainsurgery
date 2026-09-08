#!/usr/bin/env python3
"""Merge two GPT-2 MLP fine-tunes with task-vector arithmetic."""

import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
DEST = Path("out/T4/model.safetensors")
TEMP_DEST = DEST.with_suffix(".safetensors.tmp")
LAMBDA = 0.4
EXPECTED_TENSOR_COUNT = 160


def mlp_names() -> set[str]:
    suffixes = (
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    )
    return {f"h.{layer}.{suffix}" for layer in range(12) for suffix in suffixes}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    expected_mlp = mlp_names()
    require(len(expected_mlp) == 48, "internal error: expected MLP set is not 48 tensors")

    # Keep all three files open so verification and arithmetic see one stable view.
    with (
        safe_open(BASE, framework="pt", device="cpu") as base_file,
        safe_open(FT1, framework="pt", device="cpu") as ft1_file,
        safe_open(FT2, framework="pt", device="cpu") as ft2_file,
    ):
        base_keys = set(base_file.keys())
        ft1_keys = set(ft1_file.keys())
        ft2_keys = set(ft2_file.keys())

        require(
            base_keys == ft1_keys == ft2_keys,
            "checkpoint tensor names differ: "
            f"base-only={sorted(base_keys - ft1_keys - ft2_keys)}, "
            f"ft1-only={sorted(ft1_keys - base_keys)}, "
            f"ft2-only={sorted(ft2_keys - base_keys)}",
        )
        require(
            len(base_keys) == EXPECTED_TENSOR_COUNT,
            f"expected {EXPECTED_TENSOR_COUNT} input tensors, found {len(base_keys)}",
        )
        missing_mlp = expected_mlp - base_keys
        require(not missing_mlp, f"missing expected MLP tensors: {sorted(missing_mlp)}")

        # Verify the complete layouts first. In particular, no output is assembled
        # until every frozen-backbone tensor has passed exact equality checks.
        for name in sorted(base_keys):
            base_tensor = base_file.get_tensor(name)
            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            require(
                base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape,
                f"shape mismatch for {name}: "
                f"{tuple(base_tensor.shape)}, {tuple(ft1_tensor.shape)}, {tuple(ft2_tensor.shape)}",
            )
            require(
                base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype == torch.float32,
                f"dtype mismatch or non-float32 input for {name}: "
                f"{base_tensor.dtype}, {ft1_tensor.dtype}, {ft2_tensor.dtype}",
            )
            if name not in expected_mlp:
                require(
                    torch.equal(base_tensor, ft1_tensor)
                    and torch.equal(base_tensor, ft2_tensor),
                    f"frozen tensor differs from base: {name}",
                )

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_keys):
            base_tensor = base_file.get_tensor(name)
            if name in expected_mlp:
                ft1_tensor = ft1_file.get_tensor(name)
                ft2_tensor = ft2_file.get_tensor(name)
                output[name] = (
                    base_tensor
                    + LAMBDA * (ft1_tensor - base_tensor)
                    + LAMBDA * (ft2_tensor - base_tensor)
                )
                merged_count += 1
            else:
                output[name] = base_tensor.clone()

        require(merged_count == 48, f"expected to merge 48 tensors, merged {merged_count}")
        require(
            len(output) == EXPECTED_TENSOR_COUNT,
            f"expected {EXPECTED_TENSOR_COUNT} output tensors, assembled {len(output)}",
        )

    DEST.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_file(output, TEMP_DEST)
        os.replace(TEMP_DEST, DEST)
    finally:
        TEMP_DEST.unlink(missing_ok=True)

    # Reopen the persisted artifact so output-count/key checks cover serialization.
    with safe_open(DEST, framework="pt", device="cpu") as saved:
        saved_keys = set(saved.keys())
        require(
            len(saved_keys) == EXPECTED_TENSOR_COUNT,
            f"saved output has {len(saved_keys)} tensors, expected {EXPECTED_TENSOR_COUNT}",
        )
        require(saved_keys == base_keys, "saved output tensor names differ from the inputs")

    print(f"Wrote {DEST} with {len(saved_keys)} tensors; merged {merged_count} MLP tensors")


if __name__ == "__main__":
    main()
