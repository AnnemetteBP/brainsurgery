#!/usr/bin/env python3
"""Independent, exhaustive verification of the generated checkpoint."""

import json
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


base = Path("inputs/base")
output_path = Path("out/T2/model.safetensors")
with (base / "model.safetensors.index.json").open(encoding="utf-8") as handle:
    weight_map = json.load(handle)["weight_map"]

pattern = re.compile(
    r"model\.layers\.(\d+)\.self_attn\.([qkvo])_proj\.weight\Z"
)

with ExitStack() as stack:
    source_files = {
        filename: stack.enter_context(
            safe_open(base / filename, framework="pt", device="cpu")
        )
        for filename in set(weight_map.values())
    }
    generated = stack.enter_context(
        safe_open(output_path, framework="pt", device="cpu")
    )

    assert set(generated.keys()) == set(weight_map)
    assert len(generated.keys()) == 114

    for name, filename in weight_map.items():
        source = source_files[filename].get_tensor(name)
        actual = generated.get_tensor(name)
        match = pattern.fullmatch(name)
        if match:
            projection = match.group(2)
            axis = 1 if projection == "o" else 0
            expected = torch.cat(
                (source.narrow(axis, 0, 640), source.narrow(axis, 768, 1280)),
                dim=axis,
            )
        else:
            expected = source
        assert actual.dtype == expected.dtype, f"dtype mismatch: {name}"
        assert tuple(actual.shape) == tuple(expected.shape), f"shape mismatch: {name}"
        assert torch.equal(actual, expected), f"value mismatch: {name}"

print("verified all 114 tensors: keys, shapes, dtypes, and bit-exact values")
