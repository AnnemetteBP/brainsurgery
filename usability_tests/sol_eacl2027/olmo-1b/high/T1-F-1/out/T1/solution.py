#!/usr/bin/env python3
"""Remove OLMo blocks 2, 6, 10, and 14 and densely renumber the rest."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


INPUT_DIR = Path("inputs/base")
OUTPUT = Path("out/T1/model.safetensors")
TEMP_OUTPUT = OUTPUT.with_suffix(".safetensors.tmp")
REMOVED = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in REMOVED]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}
LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
Q_PROJ_KEY = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")


def input_shards() -> list[Path]:
    index_path = INPUT_DIR / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("input index has no object-valued weight_map")
    shards = sorted({INPUT_DIR / name for name in weight_map.values()})
    if not shards or any(not path.is_file() for path in shards):
        raise FileNotFoundError("one or more indexed input shards are missing")
    return shards


def build_output() -> dict[str, object]:
    source: dict[str, object] = {}
    for shard in input_shards():
        shard_tensors = load_file(str(shard), device="cpu")
        overlap = source.keys() & shard_tensors.keys()
        if overlap:
            raise ValueError(f"duplicate source keys across shards: {sorted(overlap)}")
        source.update(shard_tensors)

    if len(source) != 114:
        raise ValueError(f"expected 114 input tensors, found {len(source)}")

    layer_counts: Counter[int] = Counter()
    non_block_keys: set[str] = set()
    for key in source:
        match = LAYER_KEY.fullmatch(key)
        if match:
            layer_counts[int(match.group(1))] += 1
        else:
            non_block_keys.add(key)
    expected_counts = {i: 7 for i in range(16)}
    if dict(sorted(layer_counts.items())) != expected_counts:
        raise ValueError(f"unexpected input block structure: {dict(sorted(layer_counts.items()))}")
    expected_non_blocks = {"model.embed_tokens.weight", "lm_head.weight"}
    if non_block_keys != expected_non_blocks:
        raise ValueError(f"unexpected non-block keys: {sorted(non_block_keys)}")

    result: dict[str, object] = {}
    for old_key, tensor in source.items():
        match = LAYER_KEY.fullmatch(old_key)
        if not match:
            new_key = old_key
        else:
            old_layer = int(match.group(1))
            if old_layer in REMOVED:
                continue
            new_key = f"model.layers.{RENUMBER[old_layer]}.{match.group(2)}"
        if new_key in result:
            raise ValueError(f"rename collision at {new_key}")
        result[new_key] = tensor
    return result


def check_keys(keys: set[str]) -> None:
    if len(keys) != 86:
        raise ValueError(f"expected 86 output tensors, found {len(keys)}")

    output_layers = {
        int(match.group(1))
        for key in keys
        if (match := LAYER_KEY.fullmatch(key)) is not None
    }
    if output_layers & {12, 13, 14, 15}:
        raise ValueError(f"forbidden high-numbered blocks remain: {sorted(output_layers & {12, 13, 14, 15})}")
    if output_layers != set(range(12)):
        raise ValueError(f"expected output block indices 0..11, found {sorted(output_layers)}")

    q_proj_layers = {
        int(match.group(1))
        for key in keys
        if (match := Q_PROJ_KEY.fullmatch(key)) is not None
    }
    if q_proj_layers != set(range(12)) or len(q_proj_layers) != 12:
        raise ValueError(f"expected 12 q_proj block tensors, found layers {sorted(q_proj_layers)}")


def main() -> None:
    tensors = build_output()
    check_keys(set(tensors))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP_OUTPUT.unlink(missing_ok=True)
    try:
        save_file(tensors, str(TEMP_OUTPUT), metadata={"format": "pt"})
        with safe_open(TEMP_OUTPUT, framework="pt", device="cpu") as written:
            check_keys(set(written.keys()))
        os.replace(TEMP_OUTPUT, OUTPUT)
    except BaseException:
        TEMP_OUTPUT.unlink(missing_ok=True)
        raise

    print(f"wrote {OUTPUT} with {len(tensors)} tensors and layers 0..11")


if __name__ == "__main__":
    main()
