#!/usr/bin/env python3
"""Drop OLMo blocks 2/6/10/14 and contiguously renumber the survivors."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


INPUT_DIR = Path("inputs/base")
INDEX_PATH = INPUT_DIR / "model.safetensors.index.json"
OUTPUT_PATH = Path("out/T1/model.safetensors")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
Q_PROJ_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
DROPPED = {2, 6, 10, 14}
EXPECTED_SOURCE_LAYERS = set(range(16))
EXPECTED_OUTPUT_LAYERS = set(range(12))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_output_keys(keys: set[str]) -> None:
    layer_indices = {
        int(match.group(1))
        for key in keys
        if (match := LAYER_RE.fullmatch(key)) is not None
    }
    require(
        not (layer_indices & {12, 13, 14, 15}),
        "output still contains a tensor in block 12, 13, 14, or 15",
    )

    q_proj_indices = {
        int(match.group(1))
        for key in keys
        if (match := Q_PROJ_RE.fullmatch(key)) is not None
    }
    require(
        q_proj_indices == EXPECTED_OUTPUT_LAYERS,
        f"expected q_proj tensors for exactly blocks 0..11, got {sorted(q_proj_indices)}",
    )
    require(len(keys) == 86, f"expected exactly 86 output tensors, got {len(keys)}")


def main() -> None:
    index = json.loads(INDEX_PATH.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    require(len(weight_map) == 114, f"expected 114 indexed input tensors, got {len(weight_map)}")

    block_counts: Counter[int] = Counter()
    for key in weight_map:
        match = LAYER_RE.fullmatch(key)
        if match is not None:
            block_counts[int(match.group(1))] += 1
    require(
        set(block_counts) == EXPECTED_SOURCE_LAYERS,
        f"expected source blocks 0..15, got {sorted(block_counts)}",
    )
    require(
        all(block_counts[i] == 7 for i in EXPECTED_SOURCE_LAYERS),
        f"expected 7 tensors per source block, got {dict(sorted(block_counts.items()))}",
    )

    survivors = [i for i in range(16) if i not in DROPPED]
    old_to_new = {old: new for new, old in enumerate(survivors)}
    rename: dict[str, str] = {}
    for old_key in weight_map:
        match = LAYER_RE.fullmatch(old_key)
        if match is None:
            new_key = old_key
        else:
            old_layer = int(match.group(1))
            if old_layer in DROPPED:
                continue
            new_key = f"model.layers.{old_to_new[old_layer]}.{match.group(2)}"
        require(new_key not in rename.values(), f"destination-key collision at {new_key}")
        rename[old_key] = new_key

    validate_output_keys(set(rename.values()))

    shard_tensors = {
        shard: load_file(str(INPUT_DIR / shard), device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    output_tensors = {
        new_key: shard_tensors[weight_map[old_key]][old_key]
        for old_key, new_key in rename.items()
    }
    require(set(output_tensors) == set(rename.values()), "loaded tensor keys differ from validated mapping")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=".model-", suffix=".safetensors.tmp", dir=OUTPUT_PATH.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink()
    try:
        save_file(output_tensors, str(temp_path))
        with safe_open(temp_path, framework="pt", device="cpu") as saved:
            saved_keys = set(saved.keys())
        validate_output_keys(saved_keys)
        require(saved_keys == set(output_tensors), "saved tensor keys differ from intended keys")
        os.replace(temp_path, OUTPUT_PATH)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"wrote {OUTPUT_PATH} with {len(output_tensors)} tensors")


if __name__ == "__main__":
    main()
