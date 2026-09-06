#!/usr/bin/env python
"""T4: task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])   for the
64 MLP tensors, lambda = 0.4, computed in float32 and cast back to float16.
Every other tensor is copied unchanged from base.

Before touching anything, verifies:
  - all three checkpoints have the same 244 tensor names
  - every tensor outside the 64 MLP tensors is bit-identical across all three

Aborts loudly (raises) if any of these fail. Also asserts exactly 64 tensors
were merged and the output has exactly 244 tensors.

Uses only `torch` and `safetensors` (both in F-allowed.md) directly, rather
than a merge-toolkit config, because the required checks (shared-tensor
verification, exact merge count) are precise per-tensor assertions that are
simplest to express and to fail loudly on in plain code.
"""

import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base/model.safetensors"
FT1 = "inputs/ft1/model.safetensors"
FT2 = "inputs/ft2/model.safetensors"
OUT = "out/T4/model.safetensors"

LAMBDA = 0.4

MLP_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$")


def is_mlp_tensor(name: str) -> bool:
    return MLP_RE.match(name) is not None


def load_all(path: str) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main() -> None:
    base = load_all(BASE)
    ft1 = load_all(FT1)
    ft2 = load_all(FT2)

    # --- Step 1: shared-tensor verification. Abort loudly if it fails. ---
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        missing_in_ft1 = base_keys - ft1_keys
        missing_in_ft2 = base_keys - ft2_keys
        extra_in_ft1 = ft1_keys - base_keys
        extra_in_ft2 = ft2_keys - base_keys
        raise AssertionError(
            "Tensor name mismatch across checkpoints: "
            f"missing_in_ft1={sorted(missing_in_ft1)} "
            f"missing_in_ft2={sorted(missing_in_ft2)} "
            f"extra_in_ft1={sorted(extra_in_ft1)} "
            f"extra_in_ft2={sorted(extra_in_ft2)}"
        )

    mlp_keys = {name for name in base_keys if is_mlp_tensor(name)}
    non_mlp_keys = base_keys - mlp_keys

    mismatched = []
    for name in sorted(non_mlp_keys):
        b, o1, o2 = base[name], ft1[name], ft2[name]
        if b.shape != o1.shape or b.shape != o2.shape:
            mismatched.append(f"{name}: shape mismatch base={b.shape} ft1={o1.shape} ft2={o2.shape}")
            continue
        if b.dtype != o1.dtype or b.dtype != o2.dtype:
            mismatched.append(f"{name}: dtype mismatch base={b.dtype} ft1={o1.dtype} ft2={o2.dtype}")
            continue
        if not torch.equal(b, o1) or not torch.equal(b, o2):
            mismatched.append(f"{name}: values differ outside the MLP tensor set")
    if mismatched:
        raise AssertionError(
            "Non-MLP tensors are not identical across checkpoints "
            f"(frozen-backbone assumption violated), first issues:\n"
            + "\n".join(mismatched[:20])
        )

    if len(mlp_keys) != 64:
        raise AssertionError(f"Expected exactly 64 MLP tensors, found {len(mlp_keys)}")

    # --- Step 2/3: merge. ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in base_keys:
        if name in mlp_keys:
            b = base[name].to(torch.float32)
            o1 = ft1[name].to(torch.float32)
            o2 = ft2[name].to(torch.float32)
            merged = b + LAMBDA * (o1 - b) + LAMBDA * (o2 - b)
            out[name] = merged.to(base[name].dtype).contiguous()
            merged_count += 1
        else:
            out[name] = base[name].contiguous()

    if merged_count != 64:
        raise AssertionError(f"Expected to merge exactly 64 tensors, merged {merged_count}")
    if len(out) != 244:
        raise AssertionError(f"Expected exactly 244 output tensors, got {len(out)}")

    save_file(out, OUT)
    print(f"Wrote {OUT}: {len(out)} tensors, {merged_count} merged, {len(out) - merged_count} copied from base.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(1)
