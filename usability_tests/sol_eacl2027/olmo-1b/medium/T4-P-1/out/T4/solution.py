#!/usr/bin/env python3
"""Merge two OLMo fine-tunes with task-vector arithmetic."""

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


class ShardedCheckpoint:
    """Read tensors from indexed safetensors shards."""

    def __init__(self, directory: Path, stack: ExitStack) -> None:
        index_path = directory / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        self.weight_map = index["weight_map"]
        shard_names = sorted(set(self.weight_map.values()))
        self.shards = {
            name: stack.enter_context(
                safe_open(directory / name, framework="pt", device="cpu")
            )
            for name in shard_names
        }

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.shards[self.weight_map[name]].get_tensor(name)


def expected_mlp_names() -> set[str]:
    projections = ("gate_proj", "up_proj", "down_proj")
    return {
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(16)
        for projection in projections
    }


def main() -> None:
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"Expected 48 MLP names, constructed {len(mlp_names)}")

    with ExitStack() as stack:
        base = ShardedCheckpoint(BASE_DIR, stack)
        ft1 = stack.enter_context(safe_open(FT1_PATH, framework="pt", device="cpu"))
        ft2 = stack.enter_context(safe_open(FT2_PATH, framework="pt", device="cpu"))

        base_names = base.keys()
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        if not mlp_names <= base_names:
            raise RuntimeError(
                f"Missing expected MLP tensors: {sorted(mlp_names - base_names)}"
            )

        # Verify the frozen backbone completely before constructing any output.
        frozen_names = sorted(base_names - mlp_names)
        for name in frozen_names:
            base_tensor = base.get_tensor(name)
            ft1_tensor = ft1.get_tensor(name)
            ft2_tensor = ft2.get_tensor(name)
            if not (torch.equal(base_tensor, ft1_tensor) and torch.equal(base_tensor, ft2_tensor)):
                raise RuntimeError(f"Frozen tensor differs from base: {name}")

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base_tensor = base.get_tensor(name)
            if name not in mlp_names:
                output[name] = base_tensor
                continue

            ft1_tensor = ft1.get_tensor(name)
            ft2_tensor = ft2.get_tensor(name)
            if not (
                base_tensor.shape == ft1_tensor.shape == ft2_tensor.shape
                and base_tensor.dtype == ft1_tensor.dtype == ft2_tensor.dtype == torch.float32
            ):
                raise RuntimeError(
                    f"Incompatible MLP tensor shape or dtype for {name}: "
                    f"base=({tuple(base_tensor.shape)}, {base_tensor.dtype}), "
                    f"ft1=({tuple(ft1_tensor.shape)}, {ft1_tensor.dtype}), "
                    f"ft2=({tuple(ft2_tensor.shape)}, {ft2_tensor.dtype})"
                )
            output[name] = (
                base_tensor
                + LAMBDA * (ft1_tensor - base_tensor)
                + LAMBDA * (ft2_tensor - base_tensor)
            )
            merged_count += 1

        if merged_count != 48:
            raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")
        if len(output) != 114:
            raise RuntimeError(f"Expected 114 output tensors, got {len(output)}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_file(output, OUTPUT_PATH)

    # Reopen the artifact so the required output-count check covers serialization.
    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as saved:
        saved_names = set(saved.keys())
    if len(saved_names) != 114 or saved_names != base_names:
        raise RuntimeError(
            f"Saved output has an invalid key set ({len(saved_names)} tensors)"
        )

    print(f"Wrote {len(saved_names)} tensors to {OUTPUT_PATH}; merged {merged_count} MLP tensors")


if __name__ == "__main__":
    main()
