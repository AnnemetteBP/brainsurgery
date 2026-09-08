#!/usr/bin/env python3
"""Verify and task-vector merge the two supplied OLMo fine-tunes."""

from contextlib import ExitStack
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1_PATH = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2_PATH = ROOT / "inputs" / "ft2" / "model.safetensors"
OUTPUT_PATH = Path(__file__).resolve().parent / "model.safetensors"
LAMBDA = 0.4
EXPECTED_TENSORS = 114


def mlp_names() -> set[str]:
    return {
        f"model.layers.{layer}.mlp.{projection}_proj.weight"
        for layer in range(16)
        for projection in ("gate", "up", "down")
    }


def main() -> None:
    merge_names = mlp_names()
    if len(merge_names) != 48:
        raise RuntimeError(f"Expected 48 MLP names, constructed {len(merge_names)}")

    index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    base_names = set(weight_map)

    with ExitStack() as stack:
        base_files = {
            filename: stack.enter_context(
                safe_open(BASE_DIR / filename, framework="pt", device="cpu")
            )
            for filename in sorted(set(weight_map.values()))
        }
        ft1 = stack.enter_context(safe_open(FT1_PATH, framework="pt", device="cpu"))
        ft2 = stack.enter_context(safe_open(FT2_PATH, framework="pt", device="cpu"))

        ft1_names, ft2_names = set(ft1.keys()), set(ft2.keys())
        if not (base_names == ft1_names == ft2_names):
            raise RuntimeError(
                "Checkpoint tensor names differ: "
                f"base={len(base_names)}, ft1={len(ft1_names)}, ft2={len(ft2_names)}; "
                f"base-only={sorted(base_names - ft1_names - ft2_names)}, "
                f"ft1-only={sorted(ft1_names - base_names)}, "
                f"ft2-only={sorted(ft2_names - base_names)}"
            )
        if len(base_names) != EXPECTED_TENSORS:
            raise RuntimeError(
                f"Expected {EXPECTED_TENSORS} input tensors, found {len(base_names)}"
            )
        if not merge_names <= base_names:
            raise RuntimeError(f"Missing MLP tensors: {sorted(merge_names - base_names)}")

        def base_tensor(name: str) -> torch.Tensor:
            return base_files[weight_map[name]].get_tensor(name)

        # Complete the frozen-backbone verification before computing any merge.
        unchanged_names = base_names - merge_names
        for name in sorted(unchanged_names):
            base = base_tensor(name)
            one, two = ft1.get_tensor(name), ft2.get_tensor(name)
            if not torch.equal(base, one) or not torch.equal(base, two):
                raise RuntimeError(f"Frozen tensor differs from base: {name}")

        output: dict[str, torch.Tensor] = {}
        merged = 0
        for name in sorted(base_names):
            base = base_tensor(name)
            if name in merge_names:
                one, two = ft1.get_tensor(name), ft2.get_tensor(name)
                if base.dtype != torch.float32 or one.dtype != torch.float32 or two.dtype != torch.float32:
                    raise RuntimeError(f"MLP tensor is not float32: {name}")
                if base.shape != one.shape or base.shape != two.shape:
                    raise RuntimeError(f"MLP tensor shapes differ: {name}")
                output[name] = base + LAMBDA * (one - base) + LAMBDA * (two - base)
                merged += 1
            else:
                output[name] = base

        if merged != 48:
            raise RuntimeError(f"Expected to merge 48 tensors, merged {merged}")
        if len(output) != EXPECTED_TENSORS:
            raise RuntimeError(f"Expected 114 output tensors, built {len(output)}")

        save_file(output, OUTPUT_PATH, metadata={"format": "pt"})

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as written:
        written_names = set(written.keys())
        if written_names != base_names or len(written_names) != EXPECTED_TENSORS:
            raise RuntimeError(
                f"Written output key check failed: expected 114 exact keys, got {len(written_names)}"
            )
    print(f"Wrote {OUTPUT_PATH} with {len(base_names)} tensors ({merged} merged)")


if __name__ == "__main__":
    main()
