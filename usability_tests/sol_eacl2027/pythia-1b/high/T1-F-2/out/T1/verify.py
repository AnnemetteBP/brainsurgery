#!/usr/bin/env python3
"""Independent bit-exact verification of the generated T1 checkpoint."""

import re
from pathlib import Path

import torch
from safetensors import safe_open


SOURCE = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
LAYER_KEY = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
SURVIVORS = [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15]
OLD_TO_NEW = {old: new for new, old in enumerate(SURVIVORS)}


def mapped_key(key: str) -> str | None:
    match = LAYER_KEY.fullmatch(key)
    if match is None:
        return key
    old = int(match.group(1))
    if old not in OLD_TO_NEW:
        return None
    return f"gpt_neox.layers.{OLD_TO_NEW[old]}.{match.group(2)}"


with safe_open(SOURCE, framework="pt", device="cpu") as source, safe_open(
    OUTPUT, framework="pt", device="cpu"
) as output:
    expected = {
        destination: source_key
        for source_key in source.keys()
        if (destination := mapped_key(source_key)) is not None
    }
    assert len(expected) == 184
    assert set(output.keys()) == set(expected)

    for destination, source_key in expected.items():
        source_tensor = source.get_tensor(source_key)
        output_tensor = output.get_tensor(destination)
        assert source_tensor.shape == output_tensor.shape, destination
        assert source_tensor.dtype == output_tensor.dtype, destination
        assert torch.equal(source_tensor, output_tensor), destination

print("verified 184 tensors: exact keys, shapes, dtypes, and values")
