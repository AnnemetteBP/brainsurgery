from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_FILE = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_FILE = ROOT / "inputs" / "ft2" / "model.safetensors"
OUTPUT_FILE = ROOT / "out" / "T4" / "model.safetensors"
SCALE = 0.4


def expected_mlp_names() -> set[str]:
    projections = ("gate_proj", "up_proj", "down_proj")
    return {
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(16)
        for projection in projections
    }


def open_base_handles() -> tuple[dict[str, str], dict[str, object]]:
    index_file = BASE_DIR / "model.safetensors.index.json"
    with index_file.open("r", encoding="utf-8") as stream:
        weight_map = json.load(stream)["weight_map"]

    shard_handles = {
        shard: safe_open(BASE_DIR / shard, framework="pt", device="cpu")
        for shard in set(weight_map.values())
    }
    actual_names = {
        name for handle in shard_handles.values() for name in handle.keys()
    }
    if actual_names != set(weight_map):
        raise RuntimeError("Base index tensor names do not match its shard contents")
    return weight_map, shard_handles


def main() -> None:
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"Expected 48 MLP names, constructed {len(mlp_names)}")

    weight_map, base_handles = open_base_handles()
    ft1 = safe_open(FT1_FILE, framework="pt", device="cpu")
    ft2 = safe_open(FT2_FILE, framework="pt", device="cpu")
    base_names = set(weight_map)

    # Complete every precondition check before computing any merged tensor.
    if base_names != set(ft1.keys()) or base_names != set(ft2.keys()):
        raise RuntimeError("The three checkpoints do not have identical tensor names")
    if not mlp_names <= base_names:
        missing = sorted(mlp_names - base_names)
        raise RuntimeError(f"Expected MLP tensors are missing: {missing}")

    for name in sorted(base_names - mlp_names):
        base_tensor = base_handles[weight_map[name]].get_tensor(name)
        ft1_tensor = ft1.get_tensor(name)
        ft2_tensor = ft2.get_tensor(name)
        if not torch.equal(base_tensor, ft1_tensor):
            raise RuntimeError(f"Fine-tune 1 changed shared tensor {name}")
        if not torch.equal(base_tensor, ft2_tensor):
            raise RuntimeError(f"Fine-tune 2 changed shared tensor {name}")

    output: dict[str, torch.Tensor] = {}
    merged_count = 0
    for name in sorted(base_names):
        base_tensor = base_handles[weight_map[name]].get_tensor(name)
        if name not in mlp_names:
            output[name] = base_tensor
            continue

        ft1_tensor = ft1.get_tensor(name)
        ft2_tensor = ft2.get_tensor(name)
        if not (
            base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape
            and base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype == torch.float32
        ):
            raise RuntimeError(f"MLP tensor shape/dtype mismatch for {name}")

        merged = base_tensor.clone()
        merged.add_(ft1_tensor - base_tensor, alpha=SCALE)
        merged.add_(ft2_tensor - base_tensor, alpha=SCALE)
        output[name] = merged
        merged_count += 1

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
    if len(output) != 114:
        raise RuntimeError(f"Expected 114 output tensors, got {len(output)}")

    save_file(output, OUTPUT_FILE)
    print(f"Wrote {len(output)} tensors to {OUTPUT_FILE} ({merged_count} merged)")


if __name__ == "__main__":
    main()
