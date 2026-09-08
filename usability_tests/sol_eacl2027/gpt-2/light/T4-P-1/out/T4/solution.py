from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE_PATH = Path("inputs/base/model.safetensors")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUTPUT_PATH = Path("out/T4/model.safetensors")
LAMBDA = 0.4


def main() -> None:
    base = load_file(BASE_PATH, device="cpu")
    ft1 = load_file(FT1_PATH, device="cpu")
    ft2 = load_file(FT2_PATH, device="cpu")

    mlp_suffixes = (
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    )
    mlp_names = {f"h.{layer}.{suffix}" for layer in range(12) for suffix in mlp_suffixes}

    base_names = set(base)
    if base_names != set(ft1) or base_names != set(ft2):
        raise RuntimeError("The three checkpoints do not have identical tensor names")
    if not mlp_names <= base_names:
        missing = sorted(mlp_names - base_names)
        raise RuntimeError(f"Missing expected MLP tensors: {missing}")

    # Verify the frozen-backbone precondition completely before doing arithmetic.
    for name in sorted(base_names - mlp_names):
        if not torch.equal(base[name], ft1[name]):
            raise RuntimeError(f"Fine-tune 1 changed shared tensor {name}")
        if not torch.equal(base[name], ft2[name]):
            raise RuntimeError(f"Fine-tune 2 changed shared tensor {name}")

    output = {}
    merged_count = 0
    for name in base:
        if name in mlp_names:
            if base[name].dtype != torch.float32 or ft1[name].dtype != torch.float32 or ft2[name].dtype != torch.float32:
                raise RuntimeError(f"MLP tensor {name} is not float32 in all checkpoints")
            output[name] = base[name] + LAMBDA * (ft1[name] - base[name]) + LAMBDA * (ft2[name] - base[name])
            merged_count += 1
        else:
            output[name] = base[name]

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
    if len(output) != 160:
        raise RuntimeError(f"Expected 160 output tensors, got {len(output)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT_PATH)


if __name__ == "__main__":
    main()
