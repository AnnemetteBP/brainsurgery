from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUTPUT_PATH = Path("out/T4/model.safetensors")
LAMBDA = 0.4


def mlp_tensor_names() -> set[str]:
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
    merge_names = mlp_tensor_names()
    if len(merge_names) != 64:
        raise RuntimeError(
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

        # Complete all precondition checks before constructing or writing output.
        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        missing_mlp = merge_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")

        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)

            if (
                base_tensor.shape != ft1_tensor.shape
                or base_tensor.shape != ft2_tensor.shape
                or base_tensor.dtype != ft1_tensor.dtype
                or base_tensor.dtype != ft2_tensor.dtype
            ):
                raise RuntimeError(f"Shape or dtype mismatch for tensor {name}")

            if name not in merge_names and (
                not torch.equal(base_tensor, ft1_tensor)
                or not torch.equal(base_tensor, ft2_tensor)
            ):
                raise RuntimeError(
                    f"Frozen-backbone verification failed for non-MLP tensor {name}"
                )

        output: dict[str, torch.Tensor] = {}
        merged_count = 0

        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            if name not in merge_names:
                output[name] = base_tensor
                continue

            # Both task vectors are measured from the original float32 base.
            base_f32 = base_tensor.float()
            merged = base_f32.clone()
            merged.add_(ft1_file.get_tensor(name).float().sub_(base_f32), alpha=LAMBDA)
            merged.add_(ft2_file.get_tensor(name).float().sub_(base_f32), alpha=LAMBDA)
            output[name] = merged.to(dtype=base_tensor.dtype)
            merged_count += 1

        if merged_count != 64:
            raise RuntimeError(f"Expected to merge 64 tensors, merged {merged_count}")
        if len(output) != 244:
            raise RuntimeError(f"Expected 244 output tensors, got {len(output)}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, OUTPUT_PATH)

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved_file:
        saved_names = set(saved_file.keys())
    if len(saved_names) != 244 or saved_names != base_names:
        raise RuntimeError(
            "Saved checkpoint verification failed: expected the original 244 tensor names"
        )

    print(
        f"Saved {len(saved_names)} tensors to {OUTPUT_PATH}; "
        f"merged {merged_count} MLP tensors."
    )


if __name__ == "__main__":
    main()
