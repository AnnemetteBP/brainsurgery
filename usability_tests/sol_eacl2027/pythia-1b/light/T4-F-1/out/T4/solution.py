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


def expected_mlp_keys() -> set[str]:
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
    mlp_keys = expected_mlp_keys()
    if len(mlp_keys) != 64:
        raise RuntimeError(f"Expected 64 distinct MLP keys, generated {len(mlp_keys)}")

    with (
        safe_open(BASE_PATH, framework="pt", device="cpu") as base,
        safe_open(FT1_PATH, framework="pt", device="cpu") as ft1,
        safe_open(FT2_PATH, framework="pt", device="cpu") as ft2,
    ):
        base_keys = set(base.keys())
        ft1_keys = set(ft1.keys())
        ft2_keys = set(ft2.keys())
        if not (base_keys == ft1_keys == ft2_keys):
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base-only={sorted(base_keys - ft1_keys - ft2_keys)}, "
                f"ft1-only={sorted(ft1_keys - base_keys)}, "
                f"ft2-only={sorted(ft2_keys - base_keys)}"
            )
        if not mlp_keys <= base_keys:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(mlp_keys - base_keys)}")

        # Complete every precondition check before computing any task vector.
        for key in sorted(base_keys):
            base_tensor = base.get_tensor(key)
            ft1_tensor = ft1.get_tensor(key)
            ft2_tensor = ft2.get_tensor(key)
            if not (
                base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape
                and base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype
            ):
                raise RuntimeError(f"Shape or dtype mismatch for tensor {key}")
            if key not in mlp_keys and not (
                torch.equal(base_tensor, ft1_tensor)
                and torch.equal(base_tensor, ft2_tensor)
            ):
                raise RuntimeError(f"Frozen non-MLP tensor differs from base: {key}")

        output = {}
        merged_count = 0
        for key in sorted(base_keys):
            base_tensor = base.get_tensor(key)
            if key in mlp_keys:
                base_f32 = base_tensor.float()
                merged = base_f32.add(
                    ft1.get_tensor(key).float().sub(base_f32), alpha=LAMBDA
                ).add(
                    ft2.get_tensor(key).float().sub(base_f32), alpha=LAMBDA
                )
                output[key] = merged.to(dtype=base_tensor.dtype)
                merged_count += 1
            else:
                output[key] = base_tensor

        if merged_count != 64:
            raise RuntimeError(f"Merged {merged_count} tensors, expected exactly 64")
        if len(output) != 244:
            raise RuntimeError(f"Output has {len(output)} tensors, expected exactly 244")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, OUTPUT_PATH)

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as written:
        written_keys = set(written.keys())
        if len(written_keys) != 244 or written_keys != base_keys:
            raise RuntimeError(
                f"Written output key check failed: {len(written_keys)} tensors"
            )

    print(f"Wrote {OUTPUT_PATH} with 244 tensors; merged exactly 64 MLP tensors")


if __name__ == "__main__":
    main()
