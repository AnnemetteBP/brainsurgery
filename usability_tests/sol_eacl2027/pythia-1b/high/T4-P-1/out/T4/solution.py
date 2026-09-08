from pathlib import Path
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUTPUT_PATH = Path("out/T4/model.safetensors")
TEMP_PATH = Path("out/T4/model.safetensors.tmp")
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
    assert len(mlp_names) == 64, f"Expected 64 MLP names, got {len(mlp_names)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base_file,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1_file,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2_file,
    ):
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())

        # Complete all precondition checks before constructing or writing output.
        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "Checkpoint tensor-name sets differ: "
                f"base={len(base_names)}, ft1={len(ft1_names)}, ft2={len(ft2_names)}"
            )
        if len(base_names) != 244:
            raise RuntimeError(f"Expected 244 input tensors, got {len(base_names)}")
        missing_mlp = mlp_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")

        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)

            if base_tensor.shape != ft1_tensor.shape or base_tensor.shape != ft2_tensor.shape:
                raise RuntimeError(f"Shape mismatch for tensor {name}")
            if base_tensor.dtype != ft1_tensor.dtype or base_tensor.dtype != ft2_tensor.dtype:
                raise RuntimeError(f"Dtype mismatch for tensor {name}")
            if name in mlp_names and base_tensor.dtype != torch.float16:
                raise RuntimeError(f"Expected float16 MLP tensor {name}, got {base_tensor.dtype}")

            if name not in mlp_names:
                if not torch.equal(base_tensor, ft1_tensor):
                    raise RuntimeError(f"Fine-tune 1 changed non-MLP tensor {name}")
                if not torch.equal(base_tensor, ft2_tensor):
                    raise RuntimeError(f"Fine-tune 2 changed non-MLP tensor {name}")

        output = {}
        merged_count = 0

        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            if name not in mlp_names:
                output[name] = base_tensor
                continue

            # Keep the original base in float32 while forming both task vectors.
            base_f32 = base_tensor.float()
            task1 = ft1_file.get_tensor(name).float().sub_(base_f32)
            merged_f32 = torch.add(base_f32, task1, alpha=LAMBDA)
            del task1
            task2 = ft2_file.get_tensor(name).float().sub_(base_f32)
            merged_f32.add_(task2, alpha=LAMBDA)
            output[name] = merged_f32.to(dtype=base_tensor.dtype)
            merged_count += 1

        if merged_count != 64:
            raise RuntimeError(f"Expected to merge 64 tensors, merged {merged_count}")
        if len(output) != 244:
            raise RuntimeError(f"Expected 244 output tensors, got {len(output)}")

        save_file(output, TEMP_PATH)
        os.replace(TEMP_PATH, OUTPUT_PATH)

    # Reopen the artifact so the final count check applies to the serialized file.
    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as output_file:
        written_names = set(output_file.keys())
    if len(written_names) != 244:
        raise RuntimeError(f"Serialized output has {len(written_names)} tensors, expected 244")
    if written_names != base_names:
        raise RuntimeError("Serialized output tensor names do not match the base")

    print(f"Wrote {OUTPUT_PATH} with {len(written_names)} tensors ({merged_count} merged)")


if __name__ == "__main__":
    main()
