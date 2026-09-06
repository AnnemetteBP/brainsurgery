"""T4: task-vector merge of two OLMo-1B fine-tunes.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])  for the
48 MLP tensors; every other tensor is copied unchanged from base. Verifies
before touching anything that all three checkpoints share the same tensor
names and that every non-MLP tensor is bit-identical across all three.

Uses only `torch` and `safetensors` (both in F-allowed.md) — a plain script
is the most directly auditable route for a check-then-merge task like this.
"""

import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # sandbox root (out/T4 -> out -> root)
INPUTS = ROOT / "inputs"
OUT_DIR = HERE
LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def load_sharded(index_path: Path) -> dict[str, torch.Tensor]:
    import json

    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    handles = {f: safe_open(index_path.parent / f, framework="pt") for f in shard_files}
    return {name: handles[shard].get_tensor(name) for name, shard in weight_map.items()}


def load_flat(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main() -> None:
    base = load_sharded(INPUTS / "base" / "model.safetensors.index.json")
    ft1 = load_flat(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_flat(INPUTS / "ft2" / "model.safetensors")

    # 1. Same tensor names across all three.
    names_base, names_ft1, names_ft2 = set(base), set(ft1), set(ft2)
    if not (names_base == names_ft1 == names_ft2):
        missing_ft1 = names_base - names_ft1
        extra_ft1 = names_ft1 - names_base
        missing_ft2 = names_base - names_ft2
        extra_ft2 = names_ft2 - names_base
        sys.exit(
            "Tensor name mismatch across checkpoints.\n"
            f"ft1 missing: {sorted(missing_ft1)} extra: {sorted(extra_ft1)}\n"
            f"ft2 missing: {sorted(missing_ft2)} extra: {sorted(extra_ft2)}"
        )
    if len(names_base) != 114:
        sys.exit(f"Expected 114 tensors, got {len(names_base)}")

    mlp_names = {n for n in names_base if MLP_RE.match(n)}
    if len(mlp_names) != 48:
        sys.exit(f"Expected 48 MLP tensors, found {len(mlp_names)}: {sorted(mlp_names)}")

    non_mlp_names = names_base - mlp_names
    if len(non_mlp_names) != 66:
        sys.exit(f"Expected 66 non-MLP tensors, found {len(non_mlp_names)}")

    # Non-MLP tensors must be bit-identical across all three checkpoints.
    mismatched = []
    for name in sorted(non_mlp_names):
        b, f1, f2 = base[name], ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            mismatched.append(f"{name}: shape mismatch {b.shape} vs {f1.shape} vs {f2.shape}")
            continue
        if not torch.equal(b, f1) or not torch.equal(b, f2):
            mismatched.append(f"{name}: values differ")
    if mismatched:
        sys.exit(
            "Non-MLP tensors are not identical across checkpoints "
            "(fine-tunes are not frozen-backbone as assumed):\n" + "\n".join(mismatched)
        )

    # 2. Task-arithmetic merge for the 48 MLP tensors, each vector against
    #    the unmodified base (not against a partially-merged result).
    out: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in mlp_names:
        b = base[name].to(torch.float32)
        f1 = ft1[name].to(torch.float32)
        f2 = ft2[name].to(torch.float32)
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[name] = merged.to(base[name].dtype)
        merged_count += 1

    if merged_count != 48:
        sys.exit(f"Merged {merged_count} tensors, expected 48")

    # 3. Every other tensor is taken from base unchanged.
    for name in non_mlp_names:
        out[name] = base[name]

    if len(out) != 114:
        sys.exit(f"Output has {len(out)} tensors, expected 114")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT_DIR / "model.safetensors")
    print(f"Wrote {len(out)} tensors ({merged_count} merged) to {OUT_DIR / 'model.safetensors'}")


if __name__ == "__main__":
    main()
