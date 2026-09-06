"""T4: task-vector merge of two Pythia-1B fine-tunes.

Plain script on top of `safetensors` + `torch` (both in F-allowed.md). This
task is pure elementwise tensor arithmetic with a strict equality
precondition; a hand-rolled script makes both the precondition check and the
"exactly 64 tensors merged" check explicit and easy to verify by reading,
which is why it was chosen over mergekit's task-arithmetic YAML config (that
route hides the per-tensor selection and the "everything else must be
bit-identical" check behind library internals).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
NUM_LAYERS = 16
MLP_SUFFIXES = [
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    "mlp.dense_4h_to_h.bias",
]


def mlp_tensor_names() -> set[str]:
    names = set()
    for i in range(NUM_LAYERS):
        for suffix in MLP_SUFFIXES:
            names.add(f"gpt_neox.layers.{i}.{suffix}")
    return names


def fail(msg: str) -> None:
    raise SystemExit(f"T4 merge FAILED: {msg}")


def main() -> None:
    root = Path(__file__).resolve().parent
    task_dir = root.parent.parent  # out/T4/solution.py -> .../T4-F-2
    inputs_dir = task_dir / "inputs"

    base = load_file(inputs_dir / "base" / "model.safetensors")
    ft1 = load_file(inputs_dir / "ft1" / "model.safetensors")
    ft2 = load_file(inputs_dir / "ft2" / "model.safetensors")

    # --- Step 1: verify same key set across all three checkpoints. ---
    base_keys, ft1_keys, ft2_keys = set(base), set(ft1), set(ft2)
    if not (base_keys == ft1_keys == ft2_keys):
        only_base = base_keys - ft1_keys - ft2_keys
        only_ft1 = ft1_keys - base_keys
        only_ft2 = ft2_keys - base_keys
        fail(
            "tensor name mismatch across checkpoints: "
            f"only_in_base={sorted(only_base)[:5]} "
            f"only_in_ft1={sorted(only_ft1)[:5]} "
            f"only_in_ft2={sorted(only_ft2)[:5]}"
        )

    expected_mlp = mlp_tensor_names()
    missing_mlp = expected_mlp - base_keys
    if missing_mlp:
        fail(f"expected MLP tensors missing from base checkpoint: {sorted(missing_mlp)[:5]}")

    # --- Step 1 (cont.): every tensor outside the 64 MLP tensors must be
    # bit-identical across all three checkpoints. Abort loudly otherwise. ---
    non_mlp_keys = base_keys - expected_mlp
    mismatched = []
    for key in sorted(non_mlp_keys):
        b, f1, f2 = base[key], ft1[key], ft2[key]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append((key, "shape", b.shape, f1.shape, f2.shape))
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatched.append((key, "dtype", b.dtype, f1.dtype, f2.dtype))
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched.append((key, "value", None, None, None))
    if mismatched:
        fail(
            f"{len(mismatched)} non-MLP tensor(s) differ across checkpoints "
            f"(expected bit-identical, frozen backbone assumption violated): "
            f"{mismatched[:5]}"
        )

    # --- Step 2: merge the 64 MLP tensors via task arithmetic, each task
    # vector taken against the unmodified base (not against a running merge). ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for key in sorted(base_keys):
        if key in expected_mlp:
            b32 = base[key].to(torch.float32)
            f1_32 = ft1[key].to(torch.float32)
            f2_32 = ft2[key].to(torch.float32)
            merged32 = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
            out[key] = merged32.to(base[key].dtype)
            merged_count += 1
        else:
            out[key] = base[key]

    if merged_count != 64:
        fail(f"expected to merge exactly 64 MLP tensors, merged {merged_count}")

    if len(out) != 244:
        fail(f"expected output checkpoint with exactly 244 tensors, got {len(out)}")

    out_dir = task_dir / "out" / "T4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.safetensors"
    save_file(out, str(out_path))

    print(f"OK: wrote {out_path} with {len(out)} tensors ({merged_count} merged, "
          f"{len(out) - merged_count} copied unchanged from base).")


if __name__ == "__main__":
    sys.exit(main())
