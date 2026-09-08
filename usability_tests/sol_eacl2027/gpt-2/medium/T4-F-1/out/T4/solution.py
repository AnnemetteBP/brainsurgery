#!/usr/bin/env python3
"""Merge two GPT-2 MLP fine-tunes by task-vector arithmetic."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
FT1_PATH = ROOT / "inputs/ft1/model.safetensors"
FT2_PATH = ROOT / "inputs/ft2/model.safetensors"
OUTPUT_PATH = Path(__file__).with_name("model.safetensors")
LAMBDA = 0.4


def expected_mlp_keys() -> set[str]:
    suffixes = (
        "c_fc.weight",
        "c_fc.bias",
        "c_proj.weight",
        "c_proj.bias",
    )
    return {f"h.{layer}.mlp.{suffix}" for layer in range(12) for suffix in suffixes}


def main() -> None:
    base = load_file(BASE_PATH, device="cpu")
    ft1 = load_file(FT1_PATH, device="cpu")
    ft2 = load_file(FT2_PATH, device="cpu")

    # Complete every precondition check before computing or writing any merge.
    base_keys = set(base)
    if set(ft1) != base_keys or set(ft2) != base_keys:
        raise RuntimeError("checkpoint tensor-name sets are not identical")

    mlp_keys = expected_mlp_keys()
    if len(mlp_keys) != 48 or not mlp_keys <= base_keys:
        missing = sorted(mlp_keys - base_keys)
        raise RuntimeError(
            f"expected exactly 48 MLP tensors; missing {len(missing)}: {missing}"
        )

    for name in sorted(base_keys):
        tensors = (base[name], ft1[name], ft2[name])
        if any(t.shape != tensors[0].shape for t in tensors[1:]):
            raise RuntimeError(f"shape mismatch for {name}")
        if any(t.dtype != tensors[0].dtype for t in tensors[1:]):
            raise RuntimeError(f"dtype mismatch for {name}")
        if tensors[0].dtype != torch.float32:
            raise RuntimeError(f"{name} is {tensors[0].dtype}, expected float32")
        if name not in mlp_keys and not (
            torch.equal(tensors[0], tensors[1])
            and torch.equal(tensors[0], tensors[2])
        ):
            raise RuntimeError(f"non-MLP tensor differs from base: {name}")

    merged: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name, base_tensor in base.items():
        if name in mlp_keys:
            # Both deltas are deliberately computed against the untouched base.
            merged[name] = (
                base_tensor
                + LAMBDA * (ft1[name] - base_tensor)
                + LAMBDA * (ft2[name] - base_tensor)
            )
            merged_count += 1
        else:
            merged[name] = base_tensor

    if merged_count != 48:
        raise RuntimeError(f"merged {merged_count} tensors, expected exactly 48")
    if len(merged) != 160:
        raise RuntimeError(f"output would contain {len(merged)} tensors, expected 160")

    save_file(merged, OUTPUT_PATH)

    # Verify the persisted artifact, rather than only the in-memory mapping.
    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as output:
        output_keys = set(output.keys())
    if len(output_keys) != 160 or output_keys != base_keys:
        raise RuntimeError(
            f"persisted output has {len(output_keys)} tensors or incorrect names"
        )

    print(f"Wrote {OUTPUT_PATH} with {len(output_keys)} tensors; merged {merged_count} MLP tensors")


if __name__ == "__main__":
    main()
