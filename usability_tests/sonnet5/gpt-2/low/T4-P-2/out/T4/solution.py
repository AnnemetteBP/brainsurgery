"""
T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors (h.<i>.mlp.{c_fc,c_proj}.{weight,bias} for i in 0..11);
everything else copied unchanged from base after verifying all three
checkpoints agree outside those 48 tensors.
"""

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4

BASE_PATH = "inputs/base/model.safetensors"
FT1_PATH = "inputs/ft1/model.safetensors"
FT2_PATH = "inputs/ft2/model.safetensors"
OUT_PATH = "out/T4/model.safetensors"


def mlp_tensor_names():
    names = set()
    for i in range(12):
        names.add(f"h.{i}.mlp.c_fc.weight")
        names.add(f"h.{i}.mlp.c_fc.bias")
        names.add(f"h.{i}.mlp.c_proj.weight")
        names.add(f"h.{i}.mlp.c_proj.bias")
    return names


def main():
    base = load_file(BASE_PATH)
    ft1 = load_file(FT1_PATH)
    ft2 = load_file(FT2_PATH)

    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if not (base_keys == ft1_keys == ft2_keys):
        raise AssertionError(
            "Tensor name mismatch between checkpoints: "
            f"base={len(base_keys)} ft1={len(ft1_keys)} ft2={len(ft2_keys)}; "
            f"base^ft1={base_keys ^ ft1_keys} base^ft2={base_keys ^ ft2_keys}"
        )

    mlp_names = mlp_tensor_names()
    missing = mlp_names - base_keys
    if missing:
        raise AssertionError(f"Expected MLP tensors missing from checkpoint: {missing}")

    non_mlp_names = base_keys - mlp_names

    # Step 1: everything outside the 48 MLP tensors must be identical across
    # all three checkpoints.
    mismatched = []
    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append((name, "shape", b.shape, f1.shape, f2.shape))
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched.append((name, "value", None, None, None))

    if mismatched:
        raise AssertionError(
            f"Non-MLP tensors differ across checkpoints (expected identical): {mismatched[:10]}"
        )

    out = {}
    merged_count = 0
    for name in base_keys:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.contiguous()
            merged_count += 1
        else:
            out[name] = base[name].clone()

    if merged_count != 48:
        raise AssertionError(f"Expected exactly 48 merged tensors, got {merged_count}")

    if len(out) != 160:
        raise AssertionError(f"Expected exactly 160 output tensors, got {len(out)}")

    save_file(out, OUT_PATH)
    print(f"Wrote {OUT_PATH}: {len(out)} tensors, {merged_count} merged.")


if __name__ == "__main__":
    main()
