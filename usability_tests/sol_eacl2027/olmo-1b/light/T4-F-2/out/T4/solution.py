#!/usr/bin/env python3
"""Merge two OLMo MLP fine-tunes using task-vector arithmetic."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_FILE = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_FILE = ROOT / "inputs" / "ft2" / "model.safetensors"
OUTPUT_FILE = Path(__file__).with_name("model.safetensors")
LAMBDA = 0.4


def expected_mlp_names() -> set[str]:
    projections = ("gate_proj", "up_proj", "down_proj")
    return {
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(16)
        for projection in projections
    }


def base_weight_map() -> dict[str, Path]:
    index_file = BASE_DIR / "model.safetensors.index.json"
    with index_file.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"Missing weight_map in {index_file}")
    return {name: BASE_DIR / shard for name, shard in weight_map.items()}


def read_base(name: str, weight_map: dict[str, Path]) -> torch.Tensor:
    with safe_open(weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def main() -> None:
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"Internal error: expected 48 MLP names, got {len(mlp_names)}")

    weight_map = base_weight_map()
    base_names = set(weight_map)
    with safe_open(FT1_FILE, framework="pt", device="cpu") as ft1, safe_open(
        FT2_FILE, framework="pt", device="cpu"
    ) as ft2:
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor-name mismatch: "
                f"base={len(base_names)}, ft1={len(ft1_names)}, ft2={len(ft2_names)}"
            )
        if not mlp_names <= base_names:
            missing = sorted(mlp_names - base_names)
            raise RuntimeError(f"Missing expected MLP tensors: {missing}")

        # Complete the frozen-backbone preflight before constructing any output.
        for name in sorted(base_names - mlp_names):
            base = read_base(name, weight_map)
            one = ft1.get_tensor(name)
            two = ft2.get_tensor(name)
            if not (torch.equal(base, one) and torch.equal(base, two)):
                raise RuntimeError(f"Frozen-backbone verification failed for {name}")

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base = read_base(name, weight_map)
            if name in mlp_names:
                one = ft1.get_tensor(name)
                two = ft2.get_tensor(name)
                if base.dtype != torch.float32 or one.dtype != torch.float32 or two.dtype != torch.float32:
                    raise RuntimeError(f"Expected float32 MLP tensors for {name}")
                if base.shape != one.shape or base.shape != two.shape:
                    raise RuntimeError(f"MLP tensor shape mismatch for {name}")
                output[name] = base + LAMBDA * (one - base) + LAMBDA * (two - base)
                merged_count += 1
            else:
                output[name] = base

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
    if len(output) != 114:
        raise RuntimeError(f"Expected 114 output tensors, got {len(output)}")

    save_file(output, OUTPUT_FILE)
    with safe_open(OUTPUT_FILE, framework="pt", device="cpu") as written:
        written_names = set(written.keys())
    if written_names != base_names or len(written_names) != 114:
        raise RuntimeError("Written output failed tensor-name/count verification")
    print(f"Wrote {OUTPUT_FILE} with {len(written_names)} tensors; merged {merged_count}")


if __name__ == "__main__":
    main()
