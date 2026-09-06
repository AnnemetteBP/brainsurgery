"""
T4: Task-vector merge of two fine-tunes (OLMo-1B-0724-hf).

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])   for the
48 MLP tensors (mlp.{gate,up,down}_proj.weight for layers 0..15), computed in
float32. All other tensors are copied unchanged from base. Fails loudly if
the shared-tensor precondition or the output tensor counts don't hold.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent.parent / "inputs"
OUT_DIR = HERE

LAMBDA = 0.4
NUM_LAYERS = 16
MLP_SUFFIXES = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
MLP_PATTERN = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def load_sharded_or_single(path: Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint that is either a single .safetensors file or a
    sharded directory with an index json."""
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if not index_path.exists():
            raise RuntimeError(f"{path} is a directory but has no model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))
        tensors: dict[str, torch.Tensor] = {}
        for shard_name in shard_files:
            shard = load_file(path / shard_name)
            tensors.update(shard)
        # Sanity: every name in the index must have actually been loaded.
        missing = set(weight_map) - set(tensors)
        if missing:
            raise RuntimeError(f"{path}: index references tensors missing from shards: {missing}")
        return tensors
    elif path.is_file():
        return load_file(path)
    else:
        raise RuntimeError(f"input path does not exist: {path}")


def is_mlp_tensor(name: str) -> bool:
    m = MLP_PATTERN.match(name)
    if m is None:
        return False
    layer = int(m.group(1))
    return 0 <= layer < NUM_LAYERS


def main() -> None:
    base_path = INPUTS / "base"
    ft1_path = INPUTS / "ft1" / "model.safetensors"
    ft2_path = INPUTS / "ft2" / "model.safetensors"

    base = load_sharded_or_single(base_path)
    ft1 = load_sharded_or_single(ft1_path)
    ft2 = load_sharded_or_single(ft2_path)

    # --- Step 1: verify shared structure and identical non-MLP tensors ---
    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if base_keys != ft1_keys or base_keys != ft2_keys:
        only_in_base = base_keys - ft1_keys - ft2_keys
        only_in_ft1 = ft1_keys - base_keys
        only_in_ft2 = ft2_keys - base_keys
        raise RuntimeError(
            "Tensor name sets differ between base/ft1/ft2. "
            f"only_in_base={sorted(only_in_base)} only_in_ft1={sorted(only_in_ft1)} "
            f"only_in_ft2={sorted(only_in_ft2)}"
        )

    all_names = sorted(base_keys)
    expected_mlp_names = {
        f"model.layers.{i}.{suffix}" for i in range(NUM_LAYERS) for suffix in MLP_SUFFIXES
    }
    mlp_names = {n for n in all_names if is_mlp_tensor(n)}
    if mlp_names != expected_mlp_names:
        missing = expected_mlp_names - mlp_names
        extra = mlp_names - expected_mlp_names
        raise RuntimeError(
            f"MLP tensor set does not match expectation. missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    if len(mlp_names) != 48:
        raise RuntimeError(f"expected exactly 48 MLP tensors, found {len(mlp_names)}")

    non_mlp_names = [n for n in all_names if n not in mlp_names]

    mismatches = []
    for name in non_mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatches.append(f"{name}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
            continue
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            mismatches.append(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")
            continue
        if not torch.equal(b, f1):
            mismatches.append(f"{name}: differs between base and ft1 (expected identical)")
        elif not torch.equal(b, f2):
            mismatches.append(f"{name}: differs between base and ft2 (expected identical)")
    if mismatches:
        raise RuntimeError(
            "Precondition violated: non-MLP tensors are not identical across "
            "base/ft1/ft2:\n" + "\n".join(mismatches)
        )

    # Also verify MLP tensor shapes/dtypes agree across the three checkpoints
    # (arithmetic below assumes this).
    for name in mlp_names:
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            raise RuntimeError(f"{name}: shape mismatch base={b.shape} ft1={f1.shape} ft2={f2.shape}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            raise RuntimeError(f"{name}: dtype mismatch base={b.dtype} ft1={f1.dtype} ft2={f2.dtype}")

    # --- Step 2 & 3: merge MLP tensors, copy everything else unchanged ---
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in all_names:
        if name in mlp_names:
            b = base[name].to(torch.float32)
            f1 = ft1[name].to(torch.float32)
            f2 = ft2[name].to(torch.float32)
            merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
            out[name] = merged.to(base[name].dtype).contiguous()
            merged_count += 1
        else:
            out[name] = base[name].clone().contiguous()

    if merged_count != 48:
        raise RuntimeError(f"expected to merge exactly 48 tensors, merged {merged_count}")
    if len(out) != 114:
        raise RuntimeError(f"expected output to have exactly 114 tensors, got {len(out)}")

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "model.safetensors"
    save_file(out, str(out_file), metadata={"format": "pt"})

    # Final sanity re-read.
    check = load_file(str(out_file))
    if len(check) != 114:
        raise RuntimeError(f"output file round-trip check failed: has {len(check)} tensors, expected 114")

    print(f"Wrote {out_file} with {len(out)} tensors ({merged_count} merged, "
          f"{len(out) - merged_count} copied unchanged).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level failure must be loud
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
