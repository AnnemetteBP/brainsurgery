#!/usr/bin/env python3
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_DIR = Path("inputs/base")
FT1_PATH = Path("inputs/ft1/model.safetensors")
FT2_PATH = Path("inputs/ft2/model.safetensors")
OUTPUT_PATH = Path("out/T4/model.safetensors")
LAMBDA = 0.4


def expected_mlp_names():
    projections = ("gate_proj", "up_proj", "down_proj")
    return {
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(16)
        for projection in projections
    }


def main():
    with (BASE_DIR / "model.safetensors.index.json").open() as index_file:
        weight_map = json.load(index_file)["weight_map"]

    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"Expected 48 MLP tensor names, constructed {len(mlp_names)}")

    with ExitStack() as stack:
        base_files = {
            shard: stack.enter_context(
                safe_open(BASE_DIR / shard, framework="pt", device="cpu")
            )
            for shard in set(weight_map.values())
        }
        ft1 = stack.enter_context(safe_open(FT1_PATH, framework="pt", device="cpu"))
        ft2 = stack.enter_context(safe_open(FT2_PATH, framework="pt", device="cpu"))

        base_names = set(weight_map)
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        if not mlp_names.issubset(base_names):
            raise RuntimeError(
                f"Missing expected MLP tensors: {sorted(mlp_names - base_names)}"
            )

        def base_tensor(name):
            return base_files[weight_map[name]].get_tensor(name)

        # Verify the frozen-backbone precondition completely before merging.
        for name in sorted(base_names - mlp_names):
            base = base_tensor(name)
            first = ft1.get_tensor(name)
            second = ft2.get_tensor(name)
            if base.shape != first.shape or base.shape != second.shape:
                raise RuntimeError(f"Shape mismatch for shared tensor {name}")
            if base.dtype != first.dtype or base.dtype != second.dtype:
                raise RuntimeError(f"Dtype mismatch for shared tensor {name}")
            if not torch.equal(base, first) or not torch.equal(base, second):
                raise RuntimeError(f"Frozen shared tensor differs from base: {name}")

        output = {}
        merged_count = 0
        for name in sorted(base_names):
            base = base_tensor(name)
            if name in mlp_names:
                first = ft1.get_tensor(name)
                second = ft2.get_tensor(name)
                if base.shape != first.shape or base.shape != second.shape:
                    raise RuntimeError(f"Shape mismatch for MLP tensor {name}")
                if not (base.dtype == first.dtype == second.dtype == torch.float32):
                    raise RuntimeError(f"MLP tensor is not consistently float32: {name}")
                merged = base.clone()
                merged.add_(first - base, alpha=LAMBDA)
                merged.add_(second - base, alpha=LAMBDA)
                output[name] = merged
                merged_count += 1
            else:
                output[name] = base

        if merged_count != 48:
            raise RuntimeError(f"Merged {merged_count} tensors instead of 48")
        if len(output) != 114:
            raise RuntimeError(f"Output contains {len(output)} tensors instead of 114")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, OUTPUT_PATH)

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved:
        if len(saved.keys()) != 114:
            raise RuntimeError(
                f"Saved output contains {len(saved.keys())} tensors instead of 114"
            )
    print(f"Merged {merged_count} MLP tensors into {OUTPUT_PATH} ({len(output)} tensors total)")


if __name__ == "__main__":
    main()
