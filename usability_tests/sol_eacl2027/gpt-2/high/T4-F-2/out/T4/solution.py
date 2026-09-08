from pathlib import Path
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "base": ROOT / "inputs/base/model.safetensors",
    "ft1": ROOT / "inputs/ft1/model.safetensors",
    "ft2": ROOT / "inputs/ft2/model.safetensors",
}
OUTPUT = Path(__file__).with_name("model.safetensors")
TEMP_OUTPUT = Path(__file__).with_name("model.safetensors.tmp")
LAMBDA = 0.4


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
    assert len(mlp_names) == 48, f"expected 48 MLP names, got {len(mlp_names)}"

    # Perform every input/precondition check before constructing or writing output.
    with (
        safe_open(INPUTS["base"], framework="pt", device="cpu") as base,
        safe_open(INPUTS["ft1"], framework="pt", device="cpu") as ft1,
        safe_open(INPUTS["ft2"], framework="pt", device="cpu") as ft2,
    ):
        base_names = set(base.keys())
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        assert base_names == ft1_names == ft2_names, (
            "checkpoint tensor-name sets differ: "
            f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
            f"ft1-only={sorted(ft1_names - base_names)}, "
            f"ft2-only={sorted(ft2_names - base_names)}"
        )
        assert len(base_names) == 160, f"expected 160 input tensors, got {len(base_names)}"
        assert mlp_names <= base_names, (
            f"missing expected MLP tensors: {sorted(mlp_names - base_names)}"
        )

        for name in sorted(base_names):
            base_tensor = base.get_tensor(name)
            ft1_tensor = ft1.get_tensor(name)
            ft2_tensor = ft2.get_tensor(name)
            assert base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape, (
                f"shape mismatch for {name}: "
                f"{base_tensor.shape}, {ft1_tensor.shape}, {ft2_tensor.shape}"
            )
            assert base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype, (
                f"dtype mismatch for {name}: "
                f"{base_tensor.dtype}, {ft1_tensor.dtype}, {ft2_tensor.dtype}"
            )
            if name not in mlp_names:
                assert torch.equal(base_tensor, ft1_tensor), (
                    f"non-MLP tensor differs between base and ft1: {name}"
                )
                assert torch.equal(base_tensor, ft2_tensor), (
                    f"non-MLP tensor differs between base and ft2: {name}"
                )

        tensors: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base_tensor = base.get_tensor(name)
            if name in mlp_names:
                ft1_tensor = ft1.get_tensor(name)
                ft2_tensor = ft2.get_tensor(name)
                assert base_tensor.dtype == torch.float32, (
                    f"merged tensor is not float32: {name} ({base_tensor.dtype})"
                )
                tensors[name] = (
                    base_tensor
                    + LAMBDA * (ft1_tensor - base_tensor)
                    + LAMBDA * (ft2_tensor - base_tensor)
                )
                merged_count += 1
            else:
                tensors[name] = base_tensor

        assert merged_count == 48, f"expected to merge 48 tensors, merged {merged_count}"
        assert len(tensors) == 160, f"expected 160 output tensors, got {len(tensors)}"
        save_file(tensors, TEMP_OUTPUT)

        # Validate the serialized artifact before atomically publishing it.
        with safe_open(TEMP_OUTPUT, framework="pt", device="cpu") as output:
            output_names = set(output.keys())
            assert output_names == base_names, "serialized output tensor names changed"
            assert len(output_names) == 160, (
                f"serialized output has {len(output_names)} tensors, expected 160"
            )
            for name in sorted(base_names - mlp_names):
                assert torch.equal(output.get_tensor(name), base.get_tensor(name)), (
                    f"serialized unchanged tensor differs from base: {name}"
                )

    os.replace(TEMP_OUTPUT, OUTPUT)
    print(f"Wrote {OUTPUT} with 160 tensors; merged exactly {merged_count} MLP tensors.")


if __name__ == "__main__":
    main()
