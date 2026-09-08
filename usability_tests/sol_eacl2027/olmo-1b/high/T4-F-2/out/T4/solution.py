#!/usr/bin/env python3
"""Verify and task-vector merge the two OLMo fine-tunes."""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_PATH = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_PATH = ROOT / "inputs" / "ft2" / "model.safetensors"
OUTPUT_PATH = Path(__file__).resolve().parent / "model.safetensors"
TEMP_PATH = Path(__file__).resolve().parent / "model.safetensors.tmp"
LAMBDA = 0.4
EXPECTED_TENSOR_COUNT = 114


class Checkpoint:
    """A uniform view over one safetensors file or an indexed shard set."""

    def __init__(self, stack: ExitStack, paths: list[Path], weight_map=None):
        self._files = {
            path.name: stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
            for path in paths
        }

        actual_locations: dict[str, str] = {}
        for filename, handle in self._files.items():
            for key in handle.keys():
                if key in actual_locations:
                    raise RuntimeError(f"duplicate tensor {key!r} in checkpoint")
                actual_locations[key] = filename

        if weight_map is not None:
            if set(weight_map) != set(actual_locations):
                missing = sorted(set(weight_map) - set(actual_locations))
                extra = sorted(set(actual_locations) - set(weight_map))
                raise RuntimeError(
                    f"base index/shard key mismatch: missing={missing}, extra={extra}"
                )
            wrong = {
                key: (weight_map[key], actual_locations[key])
                for key in weight_map
                if weight_map[key] != actual_locations[key]
            }
            if wrong:
                raise RuntimeError(f"base index has incorrect shard mappings: {wrong}")
            self._locations = dict(weight_map)
        else:
            self._locations = actual_locations

    @classmethod
    def sharded(cls, stack: ExitStack, directory: Path) -> "Checkpoint":
        index_path = directory / "model.safetensors.index.json"
        with index_path.open(encoding="utf-8") as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"missing or invalid weight_map in {index_path}")
        filenames = sorted(set(weight_map.values()))
        paths = [directory / filename for filename in filenames]
        if not all(path.is_file() for path in paths):
            raise RuntimeError("one or more indexed base shards are missing")
        return cls(stack, paths, weight_map)

    @classmethod
    def single(cls, stack: ExitStack, path: Path) -> "Checkpoint":
        if not path.is_file():
            raise RuntimeError(f"checkpoint is missing: {path}")
        return cls(stack, [path])

    @property
    def keys(self) -> set[str]:
        return set(self._locations)

    def tensor(self, key: str) -> torch.Tensor:
        return self._files[self._locations[key]].get_tensor(key)


def expected_mlp_tensors() -> dict[str, tuple[int, int]]:
    expected: dict[str, tuple[int, int]] = {}
    for layer in range(16):
        prefix = f"model.layers.{layer}.mlp"
        expected[f"{prefix}.gate_proj.weight"] = (8192, 2048)
        expected[f"{prefix}.up_proj.weight"] = (8192, 2048)
        expected[f"{prefix}.down_proj.weight"] = (2048, 8192)
    if len(expected) != 48:
        raise AssertionError(f"internal error: expected 48 MLP names, got {len(expected)}")
    return expected


def verify_inputs(
    base: Checkpoint, ft1: Checkpoint, ft2: Checkpoint, mlp_shapes: dict[str, tuple[int, int]]
) -> None:
    """Complete every input/precondition check before constructing any output."""
    if base.keys != ft1.keys or base.keys != ft2.keys:
        raise RuntimeError(
            "checkpoint tensor names differ: "
            f"base_only_vs_ft1={sorted(base.keys - ft1.keys)}, "
            f"ft1_only={sorted(ft1.keys - base.keys)}, "
            f"base_only_vs_ft2={sorted(base.keys - ft2.keys)}, "
            f"ft2_only={sorted(ft2.keys - base.keys)}"
        )
    if len(base.keys) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_TENSOR_COUNT} input tensors, got {len(base.keys)}"
        )
    if not set(mlp_shapes).issubset(base.keys):
        missing = sorted(set(mlp_shapes) - base.keys)
        raise RuntimeError(f"missing expected MLP tensors: {missing}")

    for key in sorted(base.keys):
        base_tensor = base.tensor(key)
        ft1_tensor = ft1.tensor(key)
        ft2_tensor = ft2.tensor(key)
        if (
            base_tensor.shape != ft1_tensor.shape
            or base_tensor.shape != ft2_tensor.shape
            or base_tensor.dtype != ft1_tensor.dtype
            or base_tensor.dtype != ft2_tensor.dtype
        ):
            raise RuntimeError(
                f"shape/dtype mismatch for {key}: "
                f"base={tuple(base_tensor.shape)}/{base_tensor.dtype}, "
                f"ft1={tuple(ft1_tensor.shape)}/{ft1_tensor.dtype}, "
                f"ft2={tuple(ft2_tensor.shape)}/{ft2_tensor.dtype}"
            )
        if key in mlp_shapes:
            if tuple(base_tensor.shape) != mlp_shapes[key]:
                raise RuntimeError(
                    f"unexpected MLP shape for {key}: {tuple(base_tensor.shape)}"
                )
            if base_tensor.dtype != torch.float32:
                raise RuntimeError(f"MLP tensor {key} is not float32: {base_tensor.dtype}")
        elif not torch.equal(base_tensor, ft1_tensor) or not torch.equal(
            base_tensor, ft2_tensor
        ):
            raise RuntimeError(f"frozen non-MLP tensor differs from base: {key}")


def validate_saved_file(path: Path, expected_keys: set[str]) -> None:
    with safe_open(path, framework="pt", device="cpu") as saved:
        saved_keys = set(saved.keys())
    if saved_keys != expected_keys:
        raise RuntimeError(
            f"saved output key mismatch: missing={sorted(expected_keys - saved_keys)}, "
            f"extra={sorted(saved_keys - expected_keys)}"
        )
    if len(saved_keys) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"saved output must contain {EXPECTED_TENSOR_COUNT} tensors, got {len(saved_keys)}"
        )


def main() -> None:
    started = time.monotonic()
    mlp_shapes = expected_mlp_tensors()
    with ExitStack() as stack:
        base = Checkpoint.sharded(stack, BASE_DIR)
        ft1 = Checkpoint.single(stack, FT1_PATH)
        ft2 = Checkpoint.single(stack, FT2_PATH)

        # This completes the shared-backbone verification before any output
        # tensor is calculated or any output file is written.
        verify_inputs(base, ft1, ft2, mlp_shapes)
        print("Input verification passed: common names and frozen backbone are valid")

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for key in sorted(base.keys):
            base_tensor = base.tensor(key)
            if key in mlp_shapes:
                # Both deltas are explicitly taken from the untouched base.
                merged = torch.add(
                    base_tensor,
                    ft1.tensor(key) - base_tensor,
                    alpha=LAMBDA,
                )
                merged.add_(ft2.tensor(key) - base_tensor, alpha=LAMBDA)
                if merged.dtype != torch.float32:
                    raise RuntimeError(f"merged tensor {key} is not float32")
                output[key] = merged
                merged_count += 1
            else:
                output[key] = base_tensor

        if merged_count != 48:
            raise RuntimeError(f"expected exactly 48 merged tensors, got {merged_count}")
        if len(output) != EXPECTED_TENSOR_COUNT:
            raise RuntimeError(
                f"expected exactly {EXPECTED_TENSOR_COUNT} output tensors, got {len(output)}"
            )

        TEMP_PATH.unlink(missing_ok=True)
        save_file(output, TEMP_PATH, metadata={"format": "pt"})
        validate_saved_file(TEMP_PATH, base.keys)
        os.replace(TEMP_PATH, OUTPUT_PATH)

    elapsed = time.monotonic() - started
    print(
        f"Wrote {OUTPUT_PATH} with {EXPECTED_TENSOR_COUNT} tensors "
        f"({merged_count} merged) in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
