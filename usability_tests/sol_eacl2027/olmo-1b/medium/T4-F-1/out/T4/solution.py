#!/usr/bin/env python3
"""Verify and merge two OLMo MLP task vectors into their shared base."""

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
OUTPUT_FILE = Path(__file__).with_name("model.safetensors")
LAMBDA = 0.4


def expected_mlp_names() -> set[str]:
    projections = ("gate_proj", "up_proj", "down_proj")
    return {
        f"model.layers.{layer}.mlp.{projection}.weight"
        for layer in range(16)
        for projection in projections
    }


def open_base_shards(stack: ExitStack) -> dict[str, object]:
    index_path = BASE_DIR / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as stream:
        weight_map = json.load(stream)["weight_map"]

    shard_handles = {
        shard: stack.enter_context(safe_open(BASE_DIR / shard, framework="pt", device="cpu"))
        for shard in sorted(set(weight_map.values()))
    }
    actual_names = {name for handle in shard_handles.values() for name in handle.keys()}
    if actual_names != set(weight_map):
        raise RuntimeError("Base index tensor names do not match the tensors in its shards")
    return {name: shard_handles[shard] for name, shard in weight_map.items()}


def tensor(handle: object, name: str) -> torch.Tensor:
    return handle.get_tensor(name)  # type: ignore[attr-defined]


def main() -> None:
    mlp_names = expected_mlp_names()
    if len(mlp_names) != 48:
        raise RuntimeError(f"Expected exactly 48 MLP tensor names, constructed {len(mlp_names)}")

    # No output file is opened or written until every precondition is verified.
    with ExitStack() as stack:
        base_handles = open_base_shards(stack)
        ft1 = stack.enter_context(safe_open(FT1_FILE, framework="pt", device="cpu"))
        ft2 = stack.enter_context(safe_open(FT2_FILE, framework="pt", device="cpu"))

        base_names = set(base_handles)
        ft1_names = set(ft1.keys())
        ft2_names = set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor-name sets differ: "
                f"base={len(base_names)}, ft1={len(ft1_names)}, ft2={len(ft2_names)}"
            )
        if not mlp_names <= base_names:
            missing = sorted(mlp_names - base_names)
            raise RuntimeError(f"Missing expected MLP tensors: {missing}")

        unchanged_names = sorted(base_names - mlp_names)
        for name in sorted(base_names):
            base_value = tensor(base_handles[name], name)
            ft1_value = ft1.get_tensor(name)
            ft2_value = ft2.get_tensor(name)
            if not (
                base_value.shape == ft1_value.shape == ft2_value.shape
                and base_value.dtype == ft1_value.dtype == ft2_value.dtype
            ):
                raise RuntimeError(f"Shape or dtype mismatch for tensor {name}")
            if name in mlp_names:
                if base_value.dtype != torch.float32:
                    raise RuntimeError(f"MLP tensor {name} is not float32")
            elif not (torch.equal(base_value, ft1_value) and torch.equal(base_value, ft2_value)):
                raise RuntimeError(f"Frozen-backbone verification failed for tensor {name}")

        print(f"Verified identical key sets and {len(unchanged_names)} unchanged tensors")

        output: dict[str, torch.Tensor] = {}
        merged_count = 0
        for name in sorted(base_names):
            base_value = tensor(base_handles[name], name)
            if name in mlp_names:
                # Both deltas are explicitly computed against the original base.
                merged = torch.add(base_value, ft1.get_tensor(name) - base_value, alpha=LAMBDA)
                merged.add_(ft2.get_tensor(name) - base_value, alpha=LAMBDA)
                output[name] = merged
                merged_count += 1
            else:
                output[name] = base_value

        if merged_count != 48:
            raise RuntimeError(f"Merged {merged_count} tensors instead of exactly 48")
        if len(output) != 114:
            raise RuntimeError(f"Output contains {len(output)} tensors instead of exactly 114")

        save_file(output, OUTPUT_FILE, metadata={"format": "pt"})

    with safe_open(OUTPUT_FILE, framework="pt", device="cpu") as written:
        written_names = set(written.keys())
    if len(written_names) != 114 or written_names != base_names:
        raise RuntimeError("Saved output does not contain the exact 114 input tensor names")
    print(f"Merged exactly {merged_count} MLP tensors; wrote {len(written_names)} tensors to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
