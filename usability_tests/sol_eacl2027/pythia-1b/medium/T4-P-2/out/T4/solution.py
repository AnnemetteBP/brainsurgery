from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUTPUT_PATH = Path("out/T4/model.safetensors")
LAMBDA = 0.4
EXPECTED_TENSOR_COUNT = 244


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
    assert len(merge_names) == 64, (
        f"Internal error: expected 64 MLP tensor names, got {len(merge_names)}"
    )

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base_file,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1_file,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2_file,
    ):
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())

        assert base_names == ft1_names == ft2_names, (
            "Checkpoint tensor names differ: "
            f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
            f"ft1-only={sorted(ft1_names - base_names)}, "
            f"ft2-only={sorted(ft2_names - base_names)}"
        )
        missing_mlp = merge_names - base_names
        assert not missing_mlp, f"Missing expected MLP tensors: {sorted(missing_mlp)}"

        # Validate the frozen-backbone precondition completely before performing
        # any task-vector arithmetic or constructing the output state dictionary.
        for name in sorted(base_names - merge_names):
            base_tensor = base_file.get_tensor(name)
            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            assert torch.equal(base_tensor, ft1_tensor), (
                f"Fine-tune 1 changed non-MLP tensor {name}"
            )
            assert torch.equal(base_tensor, ft2_tensor), (
                f"Fine-tune 2 changed non-MLP tensor {name}"
            )

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            if name not in merge_names:
                output[name] = base_tensor
                continue

            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            assert base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape, (
                f"Shape mismatch for {name}: {base_tensor.shape}, "
                f"{ft1_tensor.shape}, {ft2_tensor.shape}"
            )
            assert base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype, (
                f"Dtype mismatch for {name}: {base_tensor.dtype}, "
                f"{ft1_tensor.dtype}, {ft2_tensor.dtype}"
            )

            base32 = base_tensor.float()
            merged = (
                base32
                + LAMBDA * (ft1_tensor.float() - base32)
                + LAMBDA * (ft2_tensor.float() - base32)
            )
            output[name] = merged.to(dtype=base_tensor.dtype)
            merged_count += 1

        assert merged_count == 64, f"Expected 64 merged tensors, got {merged_count}"
        assert len(output) == EXPECTED_TENSOR_COUNT, (
            f"Expected {EXPECTED_TENSOR_COUNT} output tensors, got {len(output)}"
        )
        assert set(output) == base_names, "Output tensor names differ from the base"

        save_file(output, OUTPUT_PATH)

    # Validate the serialized artifact, not just the in-memory dictionary.
    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved:
        saved_names = set(saved.keys())
        assert len(saved_names) == EXPECTED_TENSOR_COUNT, (
            f"Saved output has {len(saved_names)} tensors, expected "
            f"{EXPECTED_TENSOR_COUNT}"
        )
        assert saved_names == base_names, "Saved output tensor names differ from the base"

    print(f"Wrote {OUTPUT_PATH} with {len(saved_names)} tensors ({merged_count} merged)")


if __name__ == "__main__":
    main()
