#!/usr/bin/env python3
"""Independent read-back validation for the T3 artifact."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open


SOURCE = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T3")
LIMIT = 256 * 1024 * 1024
PROJECTION_SUFFIXES = (
    "attention.query_key_value.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
)
BUFFER_SUFFIXES = (
    "attention.bias",
    "attention.masked_bias",
    "attention.rotary_emb.inv_freq",
)


def expanded(suffixes: tuple[str, ...]) -> set[str]:
    return {f"gpt_neox.layers.{i}.{suffix}" for i in range(16) for suffix in suffixes}


def main() -> None:
    projections = expanded(PROJECTION_SUFFIXES)
    buffers = expanded(BUFFER_SUFFIXES)
    index = json.loads((OUTPUT / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    with safe_open(SOURCE, framework="pt", device="cpu") as source:
        expected_keys = set(source.keys()) - buffers
        assert set(weight_map) == expected_keys
        assert len(weight_map) == 196

        files = set(weight_map.values())
        actual_files = {path.name for path in OUTPUT.glob("model-*-of-*.safetensors")}
        assert files == actual_files

        dtype_counts: Counter[torch.dtype] = Counter()
        total_size = 0
        for filename in sorted(files):
            mapped = {name for name, shard in weight_map.items() if shard == filename}
            with safe_open(OUTPUT / filename, framework="pt", device="cpu") as output:
                assert set(output.keys()) == mapped
                shard_size = 0
                for name in output.keys():
                    actual = output.get_tensor(name)
                    target_dtype = torch.bfloat16 if name in projections else torch.float32
                    assert actual.dtype == target_dtype, (name, actual.dtype, target_dtype)
                    dtype_counts[actual.dtype] += 1
                    shard_size += actual.numel() * actual.element_size()

                    expected = source.get_tensor(name).to(target_dtype)
                    assert actual.shape == expected.shape
                    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), name

                assert shard_size <= LIMIT or (len(mapped) == 1 and shard_size > LIMIT)
                total_size += shard_size

    assert dtype_counts == Counter({torch.float32: 132, torch.bfloat16: 64})
    assert index["metadata"]["total_size"] == total_size
    assert not buffers.intersection(weight_map)
    print(f"Verified {len(weight_map)} tensors across {len(files)} shards; all values are bit-exact")


if __name__ == "__main__":
    main()
