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
        "c_fc.weight",
        "c_fc.bias",
        "c_proj.weight",
        "c_proj.bias",
    )
    return {f"h.{layer}.mlp.{suffix}" for layer in range(12) for suffix in suffixes}


def main() -> None:
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"internal error: expected 48 MLP names, got {len(mlp_names)}")

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2,
    ):
        base_names = set(base.keys())
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())

        # Complete every precondition check before constructing output tensors.
        if base_names != ft1_names or base_names != ft2_names:
            raise RuntimeError(
                "checkpoint tensor names differ: "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        missing_mlp = mlp_names - base_names
        if missing_mlp:
            raise RuntimeError(f"missing expected MLP tensors: {sorted(missing_mlp)}")

        unchanged_names = base_names - mlp_names
        for name in sorted(unchanged_names):
            base_tensor = base.get_tensor(name)
            if not torch.equal(base_tensor, ft1.get_tensor(name)):
                raise RuntimeError(f"fine-tune 1 changed frozen tensor {name!r}")
            if not torch.equal(base_tensor, ft2.get_tensor(name)):
                raise RuntimeError(f"fine-tune 2 changed frozen tensor {name!r}")

        output = {}
        merged_count = 0
        for name in sorted(base_names):
            base_tensor = base.get_tensor(name)
            if name in mlp_names:
                ft1_tensor = ft1.get_tensor(name)
                ft2_tensor = ft2.get_tensor(name)
                if not (
                    base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape
                    and base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype == torch.float32
                ):
                    raise RuntimeError(f"incompatible shape or dtype for MLP tensor {name!r}")
                output[name] = (
                    base_tensor
                    + LAMBDA * (ft1_tensor - base_tensor)
                    + LAMBDA * (ft2_tensor - base_tensor)
                )
                merged_count += 1
            else:
                output[name] = base_tensor

        if merged_count != 48:
            raise RuntimeError(f"merged {merged_count} tensors, expected exactly 48")
        if len(output) != 160:
            raise RuntimeError(f"output contains {len(output)} tensors, expected exactly 160")

        save_file(output, OUTPUT_PATH)

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved:
        saved_names = set(saved.keys())
        if len(saved_names) != 160:
            raise RuntimeError(f"saved file contains {len(saved_names)} tensors, expected 160")
        if saved_names != base_names:
            raise RuntimeError("saved file tensor names do not match the input tensor names")

    print(f"Wrote {OUTPUT_PATH} with 160 tensors (48 merged, 112 unchanged).")


if __name__ == "__main__":
    main()
