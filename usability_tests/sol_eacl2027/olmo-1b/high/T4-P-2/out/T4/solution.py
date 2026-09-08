#!/usr/bin/env python3
"""Merge two OLMo fine-tunes with task-vector arithmetic."""

import json
import os
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_PATH = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_PATH = ROOT / "inputs" / "ft2" / "model.safetensors"
OUTPUT_PATH = ROOT / "out" / "T4" / "model.safetensors"
TEMP_PATH = ROOT / "out" / "T4" / "model.safetensors.tmp"
SCALE = 0.4
EXPECTED_TENSOR_COUNT = 114


def expected_mlp_names():
    projections = ("gate_proj", "up_proj", "down_proj")
    return {
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(16)
        for projection in projections
    }


def open_checkpoints(stack):
    index_path = BASE_DIR / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"Missing or invalid weight_map in {index_path}")

    base_handles = {}
    discovered_base_names = set()
    for shard_name in sorted(set(weight_map.values())):
        shard_path = BASE_DIR / shard_name
        handle = stack.enter_context(
            safe_open(str(shard_path), framework="pt", device="cpu")
        )
        base_handles[shard_name] = handle
        shard_keys = set(handle.keys())
        overlap = discovered_base_names & shard_keys
        if overlap:
            raise RuntimeError(f"Duplicate base tensors across shards: {sorted(overlap)}")
        discovered_base_names.update(shard_keys)

    base_names = set(weight_map)
    if discovered_base_names != base_names:
        raise RuntimeError(
            "Base index and shard contents disagree: "
            f"missing={sorted(base_names - discovered_base_names)}, "
            f"unindexed={sorted(discovered_base_names - base_names)}"
        )
    for name, shard_name in weight_map.items():
        if name not in set(base_handles[shard_name].keys()):
            raise RuntimeError(f"Base index maps {name!r} to the wrong shard")

    ft1 = stack.enter_context(
        safe_open(str(FT1_PATH), framework="pt", device="cpu")
    )
    ft2 = stack.enter_context(
        safe_open(str(FT2_PATH), framework="pt", device="cpu")
    )
    return weight_map, base_handles, ft1, ft2


def main():
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"Expected 48 MLP names, generated {len(mlp_names)}")

    with ExitStack() as stack:
        weight_map, base_handles, ft1, ft2 = open_checkpoints(stack)

        def base_tensor(name):
            return base_handles[weight_map[name]].get_tensor(name)

        # Complete all precondition checks before creating or modifying output.
        base_names = set(weight_map)
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base_only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1_only={sorted(ft1_names - base_names)}, "
                f"ft2_only={sorted(ft2_names - base_names)}"
            )
        if len(base_names) != EXPECTED_TENSOR_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_TENSOR_COUNT} input tensors, found {len(base_names)}"
            )
        missing_mlp = mlp_names - base_names
        if missing_mlp:
            raise RuntimeError(f"Missing expected MLP tensors: {sorted(missing_mlp)}")

        for name in sorted(base_names):
            base = base_tensor(name)
            one = ft1.get_tensor(name)
            two = ft2.get_tensor(name)
            if base.shape != one.shape or base.shape != two.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"base={tuple(base.shape)}, ft1={tuple(one.shape)}, ft2={tuple(two.shape)}"
                )
            if base.dtype != one.dtype or base.dtype != two.dtype:
                raise RuntimeError(
                    f"Dtype mismatch for {name}: "
                    f"base={base.dtype}, ft1={one.dtype}, ft2={two.dtype}"
                )
            if base.dtype != torch.float32:
                raise RuntimeError(f"Expected float32 tensor {name}, found {base.dtype}")
            if name not in mlp_names:
                # Compare the underlying float32 bit patterns, including signed zero/NaN.
                if not torch.equal(base.view(torch.int32), one.view(torch.int32)):
                    raise RuntimeError(f"Frozen tensor differs between base and ft1: {name}")
                if not torch.equal(base.view(torch.int32), two.view(torch.int32)):
                    raise RuntimeError(f"Frozen tensor differs between base and ft2: {name}")

        output = {name: base_tensor(name) for name in sorted(base_names)}
        merged_count = 0
        for name in sorted(mlp_names):
            base = output[name]
            one = ft1.get_tensor(name)
            two = ft2.get_tensor(name)
            output[name] = base + SCALE * (one - base) + SCALE * (two - base)
            merged_count += 1

        if merged_count != 48:
            raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
        if len(output) != EXPECTED_TENSOR_COUNT or set(output) != base_names:
            raise RuntimeError(
                f"Output mapping must contain exactly the {EXPECTED_TENSOR_COUNT} input names"
            )

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, str(TEMP_PATH), metadata={"format": "pt"})

    try:
        with safe_open(str(TEMP_PATH), framework="pt", device="cpu") as saved:
            saved_names = set(saved.keys())
            if len(saved_names) != EXPECTED_TENSOR_COUNT or saved_names != base_names:
                raise RuntimeError(
                    f"Saved output has {len(saved_names)} tensors or an incorrect key set"
                )
        os.replace(TEMP_PATH, OUTPUT_PATH)
    except Exception:
        TEMP_PATH.unlink(missing_ok=True)
        raise

    print(f"Merged {merged_count} tensors; wrote {len(base_names)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
