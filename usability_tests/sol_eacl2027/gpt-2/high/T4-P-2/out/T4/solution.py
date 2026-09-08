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
        "c_fc.weight",
        "c_fc.bias",
        "c_proj.weight",
        "c_proj.bias",
    )
    return {f"h.{layer}.mlp.{suffix}" for layer in range(12) for suffix in suffixes}


def main() -> None:
    merge_names = mlp_tensor_names()
    if len(merge_names) != 48:
        raise RuntimeError(f"Expected 48 MLP tensor names, constructed {len(merge_names)}")

    # Validate the three checkpoints completely before constructing output tensors.
    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base_file,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1_file,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2_file,
    ):
        base_names = set(base_file.keys())
        ft1_names = set(ft1_file.keys())
        ft2_names = set(ft2_file.keys())

        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "Checkpoint tensor-name sets differ: "
                f"base_only_vs_ft1={sorted(base_names - ft1_names)}, "
                f"ft1_only_vs_base={sorted(ft1_names - base_names)}, "
                f"base_only_vs_ft2={sorted(base_names - ft2_names)}, "
                f"ft2_only_vs_base={sorted(ft2_names - base_names)}"
            )
        missing_mlp = merge_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")

        unchanged_names = base_names - merge_names
        for name in sorted(unchanged_names):
            base_tensor = base_file.get_tensor(name)
            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            if not torch.equal(base_tensor, ft1_tensor):
                raise RuntimeError(f"Frozen tensor differs between base and ft1: {name}")
            if not torch.equal(base_tensor, ft2_tensor):
                raise RuntimeError(f"Frozen tensor differs between base and ft2: {name}")

    output: dict[str, torch.Tensor] = {}
    merged_count = 0
    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base_file,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1_file,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2_file,
    ):
        for name in sorted(base_names):
            base_tensor = base_file.get_tensor(name)
            if name not in merge_names:
                output[name] = base_tensor
                continue

            ft1_tensor = ft1_file.get_tensor(name)
            ft2_tensor = ft2_file.get_tensor(name)
            if base_tensor.dtype != torch.float32:
                raise RuntimeError(f"Merged tensor is not float32: {name} ({base_tensor.dtype})")
            if ft1_tensor.dtype != torch.float32 or ft2_tensor.dtype != torch.float32:
                raise RuntimeError(f"Fine-tune tensor is not float32: {name}")
            if base_tensor.shape != ft1_tensor.shape or base_tensor.shape != ft2_tensor.shape:
                raise RuntimeError(f"Merged tensor shapes differ: {name}")

            output[name] = (
                base_tensor
                + LAMBDA * (ft1_tensor - base_tensor)
                + LAMBDA * (ft2_tensor - base_tensor)
            )
            merged_count += 1

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
    if len(output) != 160:
        raise RuntimeError(f"Expected 160 output tensors, produced {len(output)}")
    if set(output) != base_names:
        raise RuntimeError("Output tensor names differ from the base checkpoint")

    save_file(output, OUTPUT_PATH)
    print(f"Saved {len(output)} tensors to {OUTPUT_PATH}; merged {merged_count} MLP tensors")


if __name__ == "__main__":
    main()
