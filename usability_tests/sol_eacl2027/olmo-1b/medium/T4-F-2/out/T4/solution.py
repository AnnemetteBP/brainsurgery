#!/usr/bin/env python3
"""Merge two OLMo fine-tunes with task arithmetic."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_FILE = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_FILE = ROOT / "inputs" / "ft2" / "model.safetensors"
OUTPUT_FILE = Path(__file__).resolve().parent / "model.safetensors"
LAMBDA = 0.4
EXPECTED_TENSOR_COUNT = 114


def expected_mlp_names() -> set[str]:
    projections = ("gate_proj", "up_proj", "down_proj")
    return {
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(16)
        for projection in projections
    }


def load_base_index() -> dict[str, str]:
    index_path = BASE_DIR / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"Missing or invalid weight_map in {index_path}")
    return weight_map


def main() -> None:
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"Expected 48 MLP names, generated {len(mlp_names)}")

    base_map = load_base_index()
    with ExitStack() as stack:
        base_handles = {
            shard: stack.enter_context(
                safe_open(BASE_DIR / shard, framework="pt", device="cpu")
            )
            for shard in sorted(set(base_map.values()))
        }
        ft1 = stack.enter_context(safe_open(FT1_FILE, framework="pt", device="cpu"))
        ft2 = stack.enter_context(safe_open(FT2_FILE, framework="pt", device="cpu"))

        # Validate the index against the physical shards before inspecting values.
        physical_base_names: set[str] = set()
        for shard, handle in base_handles.items():
            shard_names = set(handle.keys())
            expected_in_shard = {name for name, file in base_map.items() if file == shard}
            if shard_names != expected_in_shard:
                raise RuntimeError(f"Base index/shard key mismatch in {shard}")
            if physical_base_names & shard_names:
                raise RuntimeError(f"Duplicate tensor names across base shards: {shard}")
            physical_base_names.update(shard_names)

        base_names = set(base_map)
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (physical_base_names == base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base={len(base_names)}, ft1={len(ft1_names)}, ft2={len(ft2_names)}"
            )
        if len(base_names) != EXPECTED_TENSOR_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_TENSOR_COUNT} input tensors, found {len(base_names)}"
            )
        if not mlp_names <= base_names:
            missing = sorted(mlp_names - base_names)
            raise RuntimeError(f"Missing expected MLP tensors: {missing}")

        def base_tensor(name: str) -> torch.Tensor:
            return base_handles[base_map[name]].get_tensor(name)

        # Complete the frozen-backbone precondition check before computing any merge.
        for name in sorted(base_names - mlp_names):
            base = base_tensor(name)
            one = ft1.get_tensor(name)
            two = ft2.get_tensor(name)
            if base.shape != one.shape or base.shape != two.shape:
                raise RuntimeError(f"Shared tensor shape mismatch: {name}")
            if base.dtype != one.dtype or base.dtype != two.dtype:
                raise RuntimeError(f"Shared tensor dtype mismatch: {name}")
            if not torch.equal(base, one) or not torch.equal(base, two):
                raise RuntimeError(f"Frozen-backbone verification failed: {name}")

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base = base_tensor(name)
            if name not in mlp_names:
                output[name] = base
                continue

            one = ft1.get_tensor(name)
            two = ft2.get_tensor(name)
            if base.shape != one.shape or base.shape != two.shape:
                raise RuntimeError(f"MLP tensor shape mismatch: {name}")
            if base.dtype != torch.float32 or one.dtype != torch.float32 or two.dtype != torch.float32:
                raise RuntimeError(f"MLP tensor is not float32: {name}")
            output[name] = base + LAMBDA * (one - base) + LAMBDA * (two - base)
            merged_count += 1

        if merged_count != 48:
            raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
        if len(output) != EXPECTED_TENSOR_COUNT or set(output) != base_names:
            raise RuntimeError(
                f"Output mapping invariant failed: {len(output)} tensors"
            )

        save_file(output, OUTPUT_FILE)

    # Reopen the serialized artifact: the final-file count is an enforced check too.
    with safe_open(OUTPUT_FILE, framework="pt", device="cpu") as saved:
        saved_names = set(saved.keys())
    if len(saved_names) != EXPECTED_TENSOR_COUNT or saved_names != base_names:
        raise RuntimeError(
            f"Serialized output invariant failed: {len(saved_names)} tensors"
        )
    print(f"Wrote {OUTPUT_FILE} with {len(saved_names)} tensors; merged {merged_count}")


if __name__ == "__main__":
    main()
