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
        raise RuntimeError(f"Expected 64 MLP tensor names, constructed {len(mlp_names)}")

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2,
    ):
        base_names = set(base.keys())
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        missing_mlp = mlp_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")

        # Complete the frozen-backbone precondition check before creating output.
        for name in sorted(base_names - mlp_names):
            base_tensor = base.get_tensor(name)
            ft1_tensor = ft1.get_tensor(name)
            ft2_tensor = ft2.get_tensor(name)
            if not torch.equal(base_tensor, ft1_tensor):
                raise RuntimeError(f"Fine-tune 1 changed frozen tensor: {name}")
            if not torch.equal(base_tensor, ft2_tensor):
                raise RuntimeError(f"Fine-tune 2 changed frozen tensor: {name}")

        output = {}
        merged_count = 0
        for name in sorted(base_names):
            base_tensor = base.get_tensor(name)
            if name not in mlp_names:
                output[name] = base_tensor
                continue

            ft1_tensor = ft1.get_tensor(name)
            ft2_tensor = ft2.get_tensor(name)
            if not (
                base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape
                and base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype
            ):
                raise RuntimeError(f"MLP tensor layout differs for {name}")
            base_f32 = base_tensor.float()
            merged = (
                base_f32
                + LAMBDA * (ft1_tensor.float() - base_f32)
                + LAMBDA * (ft2_tensor.float() - base_f32)
            )
            output[name] = merged.to(dtype=base_tensor.dtype)
            merged_count += 1

        if merged_count != 64:
            raise RuntimeError(f"Merged {merged_count} tensors, expected 64")
        if len(output) != 244:
            raise RuntimeError(f"Output has {len(output)} tensors, expected 244")

        save_file(output, OUTPUT_PATH)

    # Re-open the serialized artifact so the count check covers the file itself.
    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as result:
        result_names = set(result.keys())
        if len(result_names) != 244:
            raise RuntimeError(
                f"Serialized output has {len(result_names)} tensors, expected 244"
            )
        if result_names != base_names:
            raise RuntimeError("Serialized output tensor names differ from the base")

    print(f"Wrote {OUTPUT_PATH} with 244 tensors; merged exactly 64 MLP tensors")


if __name__ == "__main__":
    main()
