"""
T4: Task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
48 MLP tensors; all other tensors are copied unchanged from base.
"""

import re
import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
BASE_PATH = "inputs/base/model.safetensors"
FT1_PATH = "inputs/ft1/model.safetensors"
FT2_PATH = "inputs/ft2/model.safetensors"
OUT_PATH = "out/T4/model.safetensors"

MLP_RE = re.compile(r"^h\.\d+\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg: str) -> None:
    print(f"FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(BASE_PATH)
    ft1 = load_file(FT1_PATH)
    ft2 = load_file(FT2_PATH)

    # Step 1: verify identical key sets across all three checkpoints.
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())
    if base_keys != ft1_keys or base_keys != ft2_keys:
        fail(
            "tensor name mismatch between checkpoints: "
            f"base-only={base_keys - ft1_keys - ft2_keys}, "
            f"ft1-only={ft1_keys - base_keys}, ft2-only={ft2_keys - base_keys}"
        )

    mlp_keys = {k for k in base_keys if MLP_RE.match(k)}
    non_mlp_keys = base_keys - mlp_keys

    if len(mlp_keys) != 48:
        fail(f"expected 48 MLP tensors, found {len(mlp_keys)}: {sorted(mlp_keys)}")

    # Verify every non-MLP tensor is bit-identical (shape, dtype, values)
    # across base/ft1/ft2.
    mismatched = []
    for k in non_mlp_keys:
        b, f1, f2 = base[k], ft1[k], ft2[k]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{k}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append(f"{k}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched.append(f"{k}: values differ outside MLP tensors")
    if mismatched:
        fail("non-MLP tensors are not identical across checkpoints:\n" + "\n".join(mismatched))

    # Step 2: compute merged MLP tensors in float32.
    out = {}
    merged_count = 0
    for k in mlp_keys:
        b = base[k].to(torch.float32)
        f1 = ft1[k].to(torch.float32)
        f2 = ft2[k].to(torch.float32)
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[k] = merged.to(base[k].dtype)
        merged_count += 1

    if merged_count != 48:
        fail(f"expected to merge exactly 48 tensors, merged {merged_count}")

    # Step 3: everything else comes from base unchanged.
    for k in non_mlp_keys:
        out[k] = base[k]

    if len(out) != 160:
        fail(f"expected 160 output tensors, got {len(out)}")

    save_file(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out)} tensors ({merged_count} merged, "
          f"{len(non_mlp_keys)} unchanged).")


if __name__ == "__main__":
    main()
