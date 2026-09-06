"""T4: task-vector merge of two OLMo-1B-0724-hf fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the 48 MLP tensors
out[X] = base[X]                                                      for everything else

Verifies, before touching anything, that base/ft1/ft2 share the same tensor
names and that every non-MLP tensor is bit-identical across all three.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # sandbox root (out/T4 -> out -> root)
INPUTS = ROOT / "inputs"
OUT_DIR = ROOT / "out" / "T4"


def load_sharded_index(dir_path: Path) -> dict[str, Path]:
    """Return {tensor_name: shard_file_path} for a directory that may be
    single-file (model.safetensors) or sharded (index.json + shards)."""
    index_file = dir_path / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
        return {name: dir_path / shard for name, shard in weight_map.items()}
    single = dir_path / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no model.safetensors or index.json under {dir_path}")
    with safe_open(str(single), framework="pt") as f:
        names = list(f.keys())
    return {name: single for name in names}


def load_tensor(shard_path: Path, name: str) -> torch.Tensor:
    with safe_open(str(shard_path), framework="pt") as f:
        return f.get_tensor(name)


def main() -> None:
    base_index = load_sharded_index(INPUTS / "base")
    ft1_index = load_sharded_index(INPUTS / "ft1")
    ft2_index = load_sharded_index(INPUTS / "ft2")

    base_names = set(base_index)
    ft1_names = set(ft1_index)
    ft2_names = set(ft2_index)

    if not (base_names == ft1_names == ft2_names):
        missing_in_ft1 = base_names - ft1_names
        missing_in_ft2 = base_names - ft2_names
        extra_in_ft1 = ft1_names - base_names
        extra_in_ft2 = ft2_names - base_names
        sys.exit(
            "ABORT: tensor name sets differ across base/ft1/ft2.\n"
            f"  missing_in_ft1={sorted(missing_in_ft1)}\n"
            f"  missing_in_ft2={sorted(missing_in_ft2)}\n"
            f"  extra_in_ft1={sorted(extra_in_ft1)}\n"
            f"  extra_in_ft2={sorted(extra_in_ft2)}"
        )

    all_names = sorted(base_names)
    mlp_names = [n for n in all_names if MLP_RE.match(n)]
    non_mlp_names = [n for n in all_names if n not in set(mlp_names)]

    print(f"total tensors: {len(all_names)}; mlp candidates: {len(mlp_names)}")

    # Step 1: verify every non-MLP tensor is identical (shape, dtype, and
    # bit-exact values) across base, ft1, ft2. Abort loudly otherwise.
    mismatches = []
    for name in non_mlp_names:
        base_t = load_tensor(base_index[name], name)
        ft1_t = load_tensor(ft1_index[name], name)
        ft2_t = load_tensor(ft2_index[name], name)
        if base_t.shape != ft1_t.shape or base_t.shape != ft2_t.shape:
            mismatches.append(f"{name}: shape mismatch base={base_t.shape} ft1={ft1_t.shape} ft2={ft2_t.shape}")
            continue
        if base_t.dtype != ft1_t.dtype or base_t.dtype != ft2_t.dtype:
            mismatches.append(f"{name}: dtype mismatch base={base_t.dtype} ft1={ft1_t.dtype} ft2={ft2_t.dtype}")
            continue
        if not torch.equal(base_t, ft1_t):
            mismatches.append(f"{name}: differs between base and ft1 (expected identical, non-MLP tensor)")
        if not torch.equal(base_t, ft2_t):
            mismatches.append(f"{name}: differs between base and ft2 (expected identical, non-MLP tensor)")

    if mismatches:
        sys.exit(
            "ABORT: non-MLP tensors are not identical across base/ft1/ft2 "
            f"({len(mismatches)} mismatch(es)):\n  " + "\n  ".join(mismatches[:20])
        )

    if len(mlp_names) != 48:
        sys.exit(f"ABORT: expected exactly 48 MLP tensors to merge, found {len(mlp_names)}: {mlp_names}")

    # Step 2 & 3: compute the merge.
    out_tensors: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in all_names:
        base_t = load_tensor(base_index[name], name)
        if name in set(mlp_names):
            ft1_t = load_tensor(ft1_index[name], name).to(torch.float32)
            ft2_t = load_tensor(ft2_index[name], name).to(torch.float32)
            base_f32 = base_t.to(torch.float32)
            merged = base_f32 + LAMBDA * (ft1_t - base_f32) + LAMBDA * (ft2_t - base_f32)
            out_tensors[name] = merged.to(base_t.dtype).contiguous()
            merged_count += 1
        else:
            out_tensors[name] = base_t.contiguous()

    if merged_count != 48:
        sys.exit(f"ABORT: merged {merged_count} tensors, expected exactly 48")

    if len(out_tensors) != 114:
        sys.exit(f"ABORT: output has {len(out_tensors)} tensors, expected exactly 114")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(OUT_DIR / "model.safetensors"))
    print(f"wrote {OUT_DIR / 'model.safetensors'} with {len(out_tensors)} tensors ({merged_count} merged)")


if __name__ == "__main__":
    main()
