"""
T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
48 MLP tensors; every other tensor is copied unchanged from base.

Uses only `safetensors` and `torch` directly (no merge toolkit needed for a
transformation this small and this order-sensitive): mergekit's task-arithmetic
merge method takes a variable number of models with per-model weights, which
covers this exact formula, but doing the read/verify/compute/write by hand
keeps the "verify against the *unmodified* base" ordering explicit and
auditable, and makes the required-checks trivial to enforce directly instead
of trusting a YAML config to have the right key regexes.
"""

import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
MLP_SUFFIXES = ("mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")


def is_mlp_tensor(name: str) -> bool:
    return any(name.endswith(suf) for suf in MLP_SUFFIXES)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file("inputs/base/model.safetensors")
    ft1 = load_file("inputs/ft1/model.safetensors")
    ft2 = load_file("inputs/ft2/model.safetensors")

    # Step 1: verify identical key sets across all three checkpoints.
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        fail(
            "tensor name mismatch across checkpoints: "
            f"base-only={base_keys - ft1_keys - ft2_keys}, "
            f"ft1-only={ft1_keys - base_keys}, ft2-only={ft2_keys - base_keys}"
        )

    mlp_keys = {k for k in base_keys if is_mlp_tensor(k)}
    non_mlp_keys = base_keys - mlp_keys

    if len(mlp_keys) != 48:
        fail(f"expected exactly 48 MLP tensors, found {len(mlp_keys)}: {sorted(mlp_keys)}")

    # Step 1 (continued): every non-MLP tensor must be bit-identical across
    # all three checkpoints (frozen-backbone precondition for task arithmetic).
    mismatched = []
    for k in non_mlp_keys:
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{k}: shape mismatch base={tuple(b.shape)} "
                               f"ft1={tuple(f1.shape)} ft2={tuple(f2.shape)}")
        elif b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append(f"{k}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
        elif not (b.equal(f1) and b.equal(f2)):
            mismatched.append(f"{k}: values differ outside the MLP tensors")
    if mismatched:
        fail(
            "non-MLP tensors are not identical across base/ft1/ft2 "
            f"({len(mismatched)} mismatches):\n" + "\n".join(mismatched[:20])
        )

    # Step 2: task-arithmetic merge for the 48 MLP tensors, each vector taken
    # against the unmodified base (all three loaded fresh above, so there is
    # no risk of chaining one merge's output into the other's base).
    out = {}
    for k in non_mlp_keys:
        out[k] = base[k].clone()
    for k in mlp_keys:
        b = base[k].float()
        f1 = ft1[k].float()
        f2 = ft2[k].float()
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[k] = merged.to(base[k].dtype)

    if len(out) != 160:
        fail(f"expected exactly 160 output tensors, got {len(out)}")
    merged_count = sum(1 for k in out if is_mlp_tensor(k))
    if merged_count != 48:
        fail(f"expected exactly 48 merged tensors, got {merged_count}")

    save_file(out, "out/T4/model.safetensors")
    print(f"wrote out/T4/model.safetensors: {len(out)} tensors, {merged_count} merged")


if __name__ == "__main__":
    main()
