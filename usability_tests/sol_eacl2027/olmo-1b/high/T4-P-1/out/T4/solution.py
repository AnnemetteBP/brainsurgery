#!/usr/bin/env python3
"""Merge two OLMo fine-tunes into their common base using task arithmetic."""

import json
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
LAMBDA = 0.4

MLP_NAMES = {
    f"model.layers.{layer}.mlp.{projection}_proj.weight"
    for layer in range(16)
    for projection in ("gate", "up", "down")
}


def open_base_shards(stack: ExitStack):
    """Open all indexed base shards and return key-to-reader mappings."""
    index_path = BASE_DIR / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"Missing or invalid weight_map in {index_path}")

    readers = {}
    actual_locations = {}
    for shard_name in sorted(set(weight_map.values())):
        reader = stack.enter_context(
            safe_open(BASE_DIR / shard_name, framework="pt", device="cpu")
        )
        readers[shard_name] = reader
        for name in reader.keys():
            if name in actual_locations:
                raise RuntimeError(f"Duplicate base tensor across shards: {name}")
            actual_locations[name] = shard_name

    if set(weight_map) != set(actual_locations):
        missing = sorted(set(weight_map) - set(actual_locations))
        extra = sorted(set(actual_locations) - set(weight_map))
        raise RuntimeError(
            f"Base index/shard key mismatch; missing={missing}, extra={extra}"
        )
    for name, indexed_shard in weight_map.items():
        if actual_locations[name] != indexed_shard:
            raise RuntimeError(
                f"Base index points {name} to {indexed_shard}, "
                f"but it is stored in {actual_locations[name]}"
            )

    return weight_map, readers


def base_tensor(name, weight_map, readers):
    return readers[weight_map[name]].get_tensor(name)


def main():
    if len(MLP_NAMES) != 48:
        raise RuntimeError(f"Expected 48 MLP tensor names, generated {len(MLP_NAMES)}")

    with ExitStack() as stack:
        weight_map, base_readers = open_base_shards(stack)
        ft1 = stack.enter_context(safe_open(FT1_PATH, framework="pt", device="cpu"))
        ft2 = stack.enter_context(safe_open(FT2_PATH, framework="pt", device="cpu"))

        # Complete all precondition checks before computing or writing any result.
        base_names = set(weight_map)
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor-name mismatch: "
                f"base_only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1_only={sorted(ft1_names - base_names)}, "
                f"ft2_only={sorted(ft2_names - base_names)}"
            )
        if not MLP_NAMES.issubset(base_names):
            raise RuntimeError(
                f"Missing expected MLP tensors: {sorted(MLP_NAMES - base_names)}"
            )

        for name in sorted(base_names):
            base = base_tensor(name, weight_map, base_readers)
            one = ft1.get_tensor(name)
            two = ft2.get_tensor(name)
            if base.shape != one.shape or base.shape != two.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: base={tuple(base.shape)}, "
                    f"ft1={tuple(one.shape)}, ft2={tuple(two.shape)}"
                )
            if base.dtype != one.dtype or base.dtype != two.dtype:
                raise RuntimeError(
                    f"Dtype mismatch for {name}: base={base.dtype}, "
                    f"ft1={one.dtype}, ft2={two.dtype}"
                )
            if name in MLP_NAMES:
                if base.dtype != torch.float32:
                    raise RuntimeError(f"MLP tensor {name} is not float32: {base.dtype}")
            elif not torch.equal(base, one) or not torch.equal(base, two):
                raise RuntimeError(f"Frozen tensor differs from base: {name}")

        output = {}
        merged_count = 0
        for name in sorted(base_names):
            base = base_tensor(name, weight_map, base_readers)
            if name not in MLP_NAMES:
                output[name] = base
                continue

            # Both deltas are taken against the original, unmodified base tensor.
            delta1 = ft1.get_tensor(name) - base
            merged = torch.add(base, delta1, alpha=LAMBDA)
            del delta1
            merged.add_(ft2.get_tensor(name) - base, alpha=LAMBDA)
            output[name] = merged
            merged_count += 1

        if merged_count != 48:
            raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
        if len(output) != 114:
            raise RuntimeError(f"Expected 114 output tensors, built {len(output)}")
        if set(output) != base_names:
            raise RuntimeError("Output tensor names differ from the base checkpoint")

        save_file(output, OUTPUT_PATH)

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved:
        saved_names = set(saved.keys())
        if len(saved_names) != 114:
            raise RuntimeError(
                f"Expected 114 tensors in saved output, found {len(saved_names)}"
            )
        if saved_names != base_names:
            raise RuntimeError("Saved output tensor names differ from the base checkpoint")

    print(f"Wrote {OUTPUT_PATH} with 114 tensors; merged {merged_count} MLP tensors")


if __name__ == "__main__":
    main()
