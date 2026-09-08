#!/usr/bin/env python3
"""Merge two GPT-2 MLP fine-tunes using task-vector arithmetic."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
FT1_PATH = ROOT / "inputs/ft1/model.safetensors"
FT2_PATH = ROOT / "inputs/ft2/model.safetensors"
OUTPUT_PATH = Path(__file__).with_name("model.safetensors")
LAMBDA = 0.4
EXPECTED_TENSOR_COUNT = 160


def expected_mlp_names() -> set[str]:
    suffixes = (
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    )
    return {f"h.{layer}.{suffix}" for layer in range(12) for suffix in suffixes}


def main() -> None:
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"internal error: expected 48 MLP names, got {len(mlp_names)}")

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2,
    ):
        # Complete all input/precondition checks before constructing or saving output.
        base_names = set(base.keys())
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "checkpoint tensor-name sets differ: "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        if len(base_names) != EXPECTED_TENSOR_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_TENSOR_COUNT} input tensors, got {len(base_names)}"
            )
        missing_mlp = mlp_names - base_names
        if missing_mlp:
            raise RuntimeError(f"missing expected MLP tensors: {sorted(missing_mlp)}")

        for name in sorted(base_names):
            base_tensor = base.get_tensor(name)
            ft1_tensor = ft1.get_tensor(name)
            ft2_tensor = ft2.get_tensor(name)
            if not (
                base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape
                and base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype
            ):
                raise RuntimeError(
                    f"shape/dtype mismatch for {name}: "
                    f"base={base_tensor.shape}/{base_tensor.dtype}, "
                    f"ft1={ft1_tensor.shape}/{ft1_tensor.dtype}, "
                    f"ft2={ft2_tensor.shape}/{ft2_tensor.dtype}"
                )
            if name not in mlp_names and not (
                torch.equal(base_tensor, ft1_tensor)
                and torch.equal(base_tensor, ft2_tensor)
            ):
                raise RuntimeError(f"non-MLP tensor differs from base: {name}")

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base_tensor = base.get_tensor(name)
            if name in mlp_names:
                if base_tensor.dtype != torch.float32:
                    raise RuntimeError(
                        f"MLP tensor {name} must be float32, got {base_tensor.dtype}"
                    )
                ft1_tensor = ft1.get_tensor(name)
                ft2_tensor = ft2.get_tensor(name)
                output[name] = (
                    base_tensor
                    + LAMBDA * (ft1_tensor - base_tensor)
                    + LAMBDA * (ft2_tensor - base_tensor)
                )
                merged_count += 1
            else:
                output[name] = base_tensor

        if merged_count != 48:
            raise RuntimeError(f"expected to merge 48 tensors, merged {merged_count}")
        if len(output) != EXPECTED_TENSOR_COUNT:
            raise RuntimeError(
                f"expected {EXPECTED_TENSOR_COUNT} output tensors, got {len(output)}"
            )

        save_file(output, OUTPUT_PATH)

    # Validate the persisted artifact, including exact preservation of shared tensors.
    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base,
        safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved,
    ):
        saved_names = set(saved.keys())
        if saved_names != base_names:
            raise RuntimeError("saved output tensor-name set differs from the base")
        if len(saved_names) != EXPECTED_TENSOR_COUNT:
            raise RuntimeError(
                f"saved output has {len(saved_names)} tensors, expected {EXPECTED_TENSOR_COUNT}"
            )
        for name in sorted(base_names - mlp_names):
            if not torch.equal(base.get_tensor(name), saved.get_tensor(name)):
                raise RuntimeError(f"saved non-MLP tensor changed: {name}")

    print(
        f"Wrote {OUTPUT_PATH} with {len(saved_names)} tensors; "
        f"merged {merged_count} MLP tensors at lambda={LAMBDA}."
    )


if __name__ == "__main__":
    main()
