"""
T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])
for the 48 MLP tensors; everything else copied unchanged from base.

Plain script on top of `safetensors` + `torch` (no merge-toolkit needed:
mergekit's task-arithmetic recipe operates on full HF export directories and
would obscure the exact-precondition check this task requires, so a direct
script gives full control over verification and error classes).
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
BASE_DIR = "inputs/base"
FT1_DIR = "inputs/ft1"
FT2_DIR = "inputs/ft2"
OUT_PATH = "out/T4/model.safetensors"


def mlp_tensor_names():
    names = []
    for i in range(12):
        for suffix in ["mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"]:
            names.append(f"h.{i}.{suffix}")
    return set(names)


def load_all(path):
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    base = load_all(f"{BASE_DIR}/model.safetensors")
    ft1 = load_all(f"{FT1_DIR}/model.safetensors")
    ft2 = load_all(f"{FT2_DIR}/model.safetensors")

    # Step 1: same tensor names across all three checkpoints.
    names_base, names_ft1, names_ft2 = set(base), set(ft1), set(ft2)
    if not (names_base == names_ft1 == names_ft2):
        missing_ft1 = names_base - names_ft1
        extra_ft1 = names_ft1 - names_base
        missing_ft2 = names_base - names_ft2
        extra_ft2 = names_ft2 - names_base
        fail(
            "tensor name mismatch across checkpoints: "
            f"ft1 missing={missing_ft1} extra={extra_ft1}; "
            f"ft2 missing={missing_ft2} extra={extra_ft2}"
        )

    mlp_names = mlp_tensor_names()
    if not mlp_names.issubset(names_base):
        fail(f"expected MLP tensor names not found in base: {mlp_names - names_base}")

    # Every tensor outside the 48 MLP tensors must be bit-identical across all three.
    non_mlp_names = names_base - mlp_names
    mismatched = []
    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append((name, "shape"))
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append((name, "dtype"))
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched.append((name, "value"))
    if mismatched:
        fail(f"non-MLP tensors differ across checkpoints (frozen-backbone assumption violated): {mismatched}")

    # Step 2: merge the 48 MLP tensors.
    out = {}
    merged_count = 0
    for name in mlp_names:
        b = base[name].to(torch.float32)
        f1 = ft1[name].to(torch.float32)
        f2 = ft2[name].to(torch.float32)
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[name] = merged.to(base[name].dtype).contiguous()
        merged_count += 1

    if merged_count != 48:
        fail(f"expected to merge exactly 48 tensors, merged {merged_count}")

    # Step 3: everything else unchanged from base.
    for name in non_mlp_names:
        out[name] = base[name].contiguous()

    if len(out) != 160:
        fail(f"expected 160 output tensors, got {len(out)}")

    save_file(out, OUT_PATH)
    print(f"OK: wrote {OUT_PATH} with {len(out)} tensors ({merged_count} merged, {len(non_mlp_names)} unchanged)")


if __name__ == "__main__":
    main()
