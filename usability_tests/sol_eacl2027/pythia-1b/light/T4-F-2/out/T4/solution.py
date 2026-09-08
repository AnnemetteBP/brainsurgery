from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
PATHS = {
    "base": ROOT / "inputs/base/model.safetensors",
    "ft1": ROOT / "inputs/ft1/model.safetensors",
    "ft2": ROOT / "inputs/ft2/model.safetensors",
}
OUTPUT = Path(__file__).with_name("model.safetensors")
LAMBDA = 0.4


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
    if len(merge_names) != 64:
        raise RuntimeError(f"Expected 64 MLP names, constructed {len(merge_names)}")

    with (
        safe_open(PATHS["base"], framework="pt", device="cpu") as base,
        safe_open(PATHS["ft1"], framework="pt", device="cpu") as ft1,
        safe_open(PATHS["ft2"], framework="pt", device="cpu") as ft2,
    ):
        # Complete every precondition check before constructing or writing output.
        base_keys = set(base.keys())
        ft1_keys = set(ft1.keys())
        ft2_keys = set(ft2.keys())
        if not (base_keys == ft1_keys == ft2_keys):
            raise RuntimeError("Checkpoint tensor-name sets differ")
        if not merge_names <= base_keys:
            missing = sorted(merge_names - base_keys)
            raise RuntimeError(f"Missing expected MLP tensors: {missing}")

        for name in sorted(base_keys):
            base_tensor = base.get_tensor(name)
            ft1_tensor = ft1.get_tensor(name)
            ft2_tensor = ft2.get_tensor(name)
            if base_tensor.shape != ft1_tensor.shape or base_tensor.shape != ft2_tensor.shape:
                raise RuntimeError(f"Shape mismatch for {name}")
            if base_tensor.dtype != ft1_tensor.dtype or base_tensor.dtype != ft2_tensor.dtype:
                raise RuntimeError(f"Dtype mismatch for {name}")
            if name not in merge_names and (
                not torch.equal(base_tensor, ft1_tensor)
                or not torch.equal(base_tensor, ft2_tensor)
            ):
                raise RuntimeError(f"Frozen tensor differs from base: {name}")

        output = {}
        merged = 0
        for name in sorted(base_keys):
            base_tensor = base.get_tensor(name)
            if name in merge_names:
                base32 = base_tensor.float()
                value = (
                    base32
                    + LAMBDA * (ft1.get_tensor(name).float() - base32)
                    + LAMBDA * (ft2.get_tensor(name).float() - base32)
                )
                output[name] = value.to(base_tensor.dtype)
                merged += 1
            else:
                output[name] = base_tensor

    if merged != 64:
        raise RuntimeError(f"Merged {merged} tensors instead of 64")
    if len(output) != 244:
        raise RuntimeError(f"Output contains {len(output)} tensors instead of 244")

    save_file(output, OUTPUT)

    with safe_open(OUTPUT, framework="pt", device="cpu") as result:
        if len(result.keys()) != 244 or set(result.keys()) != base_keys:
            raise RuntimeError("Saved output tensor set/count is incorrect")
    print(f"Wrote {OUTPUT} with {len(output)} tensors; merged {merged} MLP tensors")


if __name__ == "__main__":
    main()
