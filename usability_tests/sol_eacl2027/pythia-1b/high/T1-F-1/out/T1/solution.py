#!/usr/bin/env python3
"""Drop Pythia blocks 2/6/10/14 and compact the remaining layer names."""

from __future__ import annotations

import os
import re
from pathlib import Path

import safetensors
import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
STAGED_OUTPUT = Path("out/T1/model.safetensors.tmp")
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
QKV_SUFFIX = "attention.query_key_value.weight"
DROP = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROP]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def build_key_map(input_keys: list[str]) -> dict[str, str]:
    """Return output-key -> input-key after validating the source layout."""
    layer_counts = {i: 0 for i in range(16)}
    non_block_count = 0
    key_map: dict[str, str] = {}

    for source_key in input_keys:
        match = LAYER_RE.fullmatch(source_key)
        if match is None:
            non_block_count += 1
            destination_key = source_key
        else:
            old_layer = int(match.group(1))
            if old_layer not in layer_counts:
                raise ValueError(f"unexpected source layer index {old_layer}: {source_key}")
            layer_counts[old_layer] += 1
            if old_layer in DROP:
                continue
            destination_key = f"gpt_neox.layers.{RENUMBER[old_layer]}.{match.group(2)}"

        if destination_key in key_map:
            raise ValueError(
                f"rename collision at {destination_key!r}: "
                f"{key_map[destination_key]!r} and {source_key!r}"
            )
        key_map[destination_key] = source_key

    if len(input_keys) != 244:
        raise ValueError(f"expected 244 input tensors, found {len(input_keys)}")
    if non_block_count != 4:
        raise ValueError(f"expected 4 non-block tensors, found {non_block_count}")
    bad_counts = {i: count for i, count in layer_counts.items() if count != 15}
    if bad_counts:
        raise ValueError(f"expected 15 tensors in every input block; mismatches: {bad_counts}")

    validate_output_keys(set(key_map))
    return key_map


def validate_output_keys(keys: set[str]) -> None:
    block_indices: set[int] = set()
    qkv_count = 0
    for key in keys:
        match = LAYER_RE.fullmatch(key)
        if match is not None:
            block_indices.add(int(match.group(1)))
            if match.group(2) == QKV_SUFFIX:
                qkv_count += 1

    expected_indices = set(range(12))
    if block_indices != expected_indices:
        raise ValueError(
            f"output block indices must be 0..11, found {sorted(block_indices)}"
        )
    if block_indices & {12, 13, 14, 15}:
        raise ValueError("output still contains a tensor in block 12, 13, 14, or 15")
    if qkv_count != 12:
        raise ValueError(f"expected 12 QKV weight tensors, found {qkv_count}")
    if len(keys) != 184:
        raise ValueError(f"expected 184 output tensors, found {len(keys)}")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    STAGED_OUTPUT.unlink(missing_ok=True)

    try:
        with safe_open(INPUT, framework="pt", device="cpu") as source:
            key_map = build_key_map(list(source.keys()))
            tensors: dict[str, torch.Tensor] = {
                destination: source.get_tensor(source_key)
                for destination, source_key in key_map.items()
            }

        save_file(tensors, STAGED_OUTPUT)

        # Reopen the serialized artifact and enforce all required checks before
        # making it visible at the requested final path.
        with safe_open(STAGED_OUTPUT, framework="pt", device="cpu") as result:
            result_keys = set(result.keys())
        validate_output_keys(result_keys)
        if result_keys != set(key_map):
            raise ValueError("serialized output key set differs from the validated key map")

        os.replace(STAGED_OUTPUT, OUTPUT)
        print(f"wrote {OUTPUT} with {len(key_map)} tensors and layers 0..11")
        print(f"safetensors {safetensors.__version__}; torch {torch.__version__}")
    except BaseException:
        STAGED_OUTPUT.unlink(missing_ok=True)
        OUTPUT.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
