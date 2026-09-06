#!/usr/bin/env python
"""T4: task-vector merge of two OLMo-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the 48 MLP tensors
out[X] = base[X]                                                      for everything else

Uses plain torch + safetensors (no mergekit) because the task needs custom
three-way equality verification and an exact 114/48-tensor accounting that a
generic merge config does not expose directly; a short explicit script makes
every check auditable.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # out/T4 -> out -> sandbox root
INPUTS = REPO_ROOT / "inputs"
OUT_DIR = HERE
LAMBDA = 0.4

MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def load_base(base_dir: Path) -> dict[str, torch.Tensor]:
    index = json.loads((base_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    shards = {name: load_file(str(base_dir / name)) for name in shard_names}
    tensors = {}
    for key, shard_name in weight_map.items():
        tensors[key] = shards[shard_name][key]
    return tensors


def is_mlp_tensor(name: str) -> bool:
    m = MLP_RE.match(name)
    if m is None:
        return False
    layer = int(m.group(1))
    return 0 <= layer <= 15


def main() -> int:
    base = load_base(INPUTS / "base")
    ft1 = load_file(str(INPUTS / "ft1" / "model.safetensors"))
    ft2 = load_file(str(INPUTS / "ft2" / "model.safetensors"))

    # --- Step 1: verify same tensor names across all three checkpoints. ---
    names_base, names_ft1, names_ft2 = set(base), set(ft1), set(ft2)
    if not (names_base == names_ft1 == names_ft2):
        missing_ft1 = names_base - names_ft1
        extra_ft1 = names_ft1 - names_base
        missing_ft2 = names_base - names_ft2
        extra_ft2 = names_ft2 - names_base
        raise AssertionError(
            "tensor name mismatch across checkpoints: "
            f"ft1 missing={sorted(missing_ft1)} extra={sorted(extra_ft1)}; "
            f"ft2 missing={sorted(missing_ft2)} extra={sorted(extra_ft2)}"
        )
    if len(names_base) != 114:
        raise AssertionError(f"expected 114 tensors in base, found {len(names_base)}")

    mlp_names = {n for n in names_base if is_mlp_tensor(n)}
    if len(mlp_names) != 48:
        raise AssertionError(f"expected 48 MLP tensors, found {len(mlp_names)}: {sorted(mlp_names)}")
    non_mlp_names = names_base - mlp_names

    # --- Step 1 (cont.): every non-MLP tensor must be identical (shape, dtype, values)
    #     across all three checkpoints. Abort loudly otherwise. ---
    mismatches = []
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.dtype != f1.dtype:
            mismatches.append(f"{name}: base {b.shape}/{b.dtype} vs ft1 {f1.shape}/{f1.dtype}")
            continue
        if b.shape != f2.shape or b.dtype != f2.dtype:
            mismatches.append(f"{name}: base {b.shape}/{b.dtype} vs ft2 {f2.shape}/{f2.dtype}")
            continue
        if not torch.equal(b, f1):
            mismatches.append(f"{name}: base != ft1 (should be untouched by fine-tuning)")
        elif not torch.equal(b, f2):
            mismatches.append(f"{name}: base != ft2 (should be untouched by fine-tuning)")
    if mismatches:
        raise AssertionError(
            "non-MLP tensors differ across checkpoints, frozen-backbone assumption violated:\n"
            + "\n".join(mismatches)
        )

    # Also sanity-check shape/dtype agreement on the MLP tensors themselves.
    for name in sorted(mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if not (b.shape == f1.shape == f2.shape):
            raise AssertionError(f"{name}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
        if not (b.dtype == f1.dtype == f2.dtype):
            raise AssertionError(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")

    # --- Step 2: task-vector merge, each vector taken against the unmodified base. ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in names_base:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.to(base[name].dtype).contiguous()
            merged_count += 1
        else:
            # Step 3: everything else comes from base, unchanged.
            out[name] = base[name].contiguous()

    if merged_count != 48:
        raise AssertionError(f"expected to merge exactly 48 tensors, merged {merged_count}")
    if len(out) != 114:
        raise AssertionError(f"output has {len(out)} tensors, expected 114")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_DIR / "model.safetensors"))
    print(f"wrote {OUT_DIR / 'model.safetensors'} with {len(out)} tensors ({merged_count} merged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
